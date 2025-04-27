import numpy as np
import torch
import torch.nn as nn
import random

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from random import choice, sample
from coinche.utils import convert_cards_to_vector, convert_index_to_cards, decode_bid_action, encode_bid_action
from coinche.exceptions import PlayException
from coinche.card import Card, Suit
from coinche.trick import Trick

class Player:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.cards = []
        self.attacker = None
        self.has_belote = False
        self.self_current_score = 0
        self.opponent_current_score = 0

    def add_cards(self, cards):
        self.cards += cards

    def has_card(self, card):
        return card in self.cards

    def has_suit(self, suit):
        return any(card.suit == suit for card in self.cards)

    def get_suit_card(self, suit):
        return [card for card in self.cards if card.suit == suit]

    def remove_card(self, card):
        self.cards.remove(card)   

    @abstractmethod
    def bid(
        self,
        obs: Dict[str, Union[np.ndarray, float]],
        valid_bids: List[int],
        suits_order: List[Suit],
    ) -> int:
        """
        Choose a bid action given observation and valid bids.

        :param obs: Current game observation.
        :param valid_bids: List of valid bid action indices.
        :param suits_order: Ordering of suits for indexing.
        :return: Selected bid action index.
        """
        ...

    @abstractmethod
    def play_trick(
        self,
        trick: Trick,
        obs: Dict[str, Union[np.ndarray, float]],
        suits_order: List[Suit],
    ) -> None:
        """
        Play a card on the given trick.

        :param trick: Current trick instance.
        :param obs: Current game observation.
        :param suits_order: Ordering of suits for indexing.
        """
        ...


class RandomPlayer(Player):
    def bid(self, obs, valid_bids, suits_order):
        return random.choice(valid_bids)
    
    def play_trick(self, trick, obs, suits_order):
        legal_cards = [
            card for card in self.cards
            if trick._assert_valid_play_TrueFalse(card, self)
        ]
        if not legal_cards:
            raise PlayException("No valid cards to play - RandomPlayer")

        card = random.choice(legal_cards)
        trick.add_card(card, self)
        self.remove_card(card)

class DeterministicPlayer(RandomPlayer):
    def bid(self, obs, valid_bids, suits_order):
        cards = self.cards
        suit_counts = {suit: 0 for suit in suits_order}
        has_jack = {}
        has_nine = {}
        has_ace = {}
        bid_action = 0

        for card in cards:
            suit_counts[card.suit] += 1
            if card.rank.name == "JACK":
                has_jack[card.suit] = True
            if card.rank.name == "NINE":
                has_nine[card.suit] = True
            if card.rank.name == "ACE":
                has_ace[card.suit] = True

        last_partner_bid = int(obs[35])
        last_max_opponent_bid = int(max(
            0 if obs[34] == 43 else obs[34],
            0 if obs[36] == 43 else obs[36]
        )) # ignore the padding bid, and take the largest opponent bid

        if (last_partner_bid in [0, 43]) and (
            last_max_opponent_bid in [0, 1, 2, 3, 4, 43] # either pass, pad, or 80
        ):
            opponents_bidded_80 = last_max_opponent_bid in [1, 2, 3, 4]
            for suit in suits_order:
                j = has_jack.get(suit, False)
                n = has_nine.get(suit, False)
                count = suit_counts[suit]
                ace_else = any(s != suit and has_ace.get(s, False) for s in suits_order)
                if j and n and (count >= 1 or ace_else):
                    bid_action = encode_bid_action(90 + opponents_bidded_80 * 10, suit)
                elif (j or n) and (count >= 2 or (count >= 1 and ace_else)):
                    bid_action = encode_bid_action(80 + opponents_bidded_80 * 10, suit)

        elif (last_partner_bid in range(1,9)): # either 80 or 90
            opp_bid = decode_bid_action(last_max_opponent_bid)
            opp_value = opp_bid[0] if isinstance(opp_bid, tuple) else 0 # exclude pass, coinche, surcoinche
            value, suit = decode_bid_action(last_partner_bid)
            
            value_add = 0
            if has_nine.get(suit, False):
                value_add += 10
            if has_jack.get(suit, False):
                value_add += 20
            for s in suits_order:
                if s != suit and has_ace.get(s, False):
                    value_add += 10

            new_value = value + value_add
            if new_value > opp_value and new_value > value:
                bid_action = encode_bid_action(new_value, suit)
        return bid_action if bid_action in valid_bids else 0
    
    def play_trick(self, trick, obs, suits_order):
        legal_cards = [
            card for card in self.cards
            if trick._assert_valid_play_TrueFalse(card, self)
        ]
        if not legal_cards:
            raise PlayException("No valid cards to play - DeterministicPlayer")

        # sort by: first preferring cards in trump suit, then by rank
        legal_cards.sort(
            key=lambda c: (c.suit != trick.atout_suit, c.rank.value),
            reverse=True
        )
        card = legal_cards[0]

        trick.add_card(card, self)
        self.remove_card(card)


class SharedPolicy(nn.Module):
    def __init__(self, obs_dim=140, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class BidHead(nn.Module):
    def __init__(self, hidden_dim=128, n_actions=44):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.net(x)

class TrickHead(nn.Module):
    def __init__(self, hidden_dim=128, n_actions=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.net(x)


class AIPlayer(Player):
    def __init__(self, trick_model_path=None, bid_model_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shared    = SharedPolicy().to(self.device)
        self.bid_head  = BidHead().to(self.device)
        self.trick_head= TrickHead().to(self.device)

        # Load weights if provided
        if trick_model_path:
            state = torch.load(trick_model_path, map_location=self.device)
            self.trick_head.load_state_dict(state["trick_head"])
            self.shared.load_state_dict(state["shared"], strict=False)
        if bid_model_path:
            state = torch.load(bid_model_path, map_location=self.device)
            self.bid_head.load_state_dict(state["bid_head"])
            self.shared.load_state_dict(state["shared"], strict=False)

        self.shared.eval()
        self.bid_head.eval()
        self.trick_head.eval()

    def bid(self, obs, valid_bids, suits_order):
        # obs: dict, valid_bids: list
        vec = np.concatenate([
            obs["bids"],                   # (37,)
            obs["played_cards"],           # (32,)
            obs["player_cards"],           # (32,)
            obs["trick_cards"],            # (32,)
            obs["current_scores"] / 162.0, # (2,) normalize
            obs["extra"],                  # (5,)
        ], axis=0).astype(np.float32)      # total dim = 140

        t = torch.from_numpy(vec).to(self.device).unsqueeze(0)
        with torch.no_grad():
            features = self.shared(t)
            probs = self.bid_head(features).squeeze(0) # (44,)

        mask = torch.zeros_like(probs)
        mask[valid_bids] = 1.0
        masked = probs * mask

        if masked.sum() > 0:
            dist = masked / masked.sum()
            action = int(torch.multinomial(dist, 1).item())
        else:
            action = random.choice(valid_bids)
        return action

    def play_trick(self, trick, obs, suits_order):
        vec = np.concatenate([
            obs["bids"], 
            obs["played_cards"], 
            obs["player_cards"],
            obs["trick_cards"], 
            obs["current_scores"]/162.0, 
            obs["extra"]
        ], axis=0).astype(np.float32)

        t = torch.from_numpy(vec).to(self.device).unsqueeze(0)
        with torch.no_grad():
            feats = self.shared(t)
            probs = self.trick_head(feats).squeeze(0)  # (32,)

        # mask to only your actual cards
        valid_mask = torch.tensor([
            trick._assert_valid_play_TrueFalse(card, self) for card in self.cards
        ], dtype=torch.bool, device=self.device)
        masked = probs * valid_mask

        if masked.sum() > 0:
            dist = torch.softmax(masked, dim=-1)
            idx = int(torch.multinomial(dist, 1).item())
        else:
            raise PlayException("No valid cards to play")
        
        card = convert_index_to_cards(idx, suits_order)[0]
        trick.add_card(card, self)
        self.remove_card(card)