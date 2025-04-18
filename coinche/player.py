import numpy as np
import torch
import torch.nn as nn
import random

from random import choice, sample
from coinche.utils import convert_cards_to_vector, convert_index_to_cards, decode_bid_action, encode_bid_action
from coinche.exceptions import PlayException
from coinche.card import Card

class Player:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.cards = []
        self.attacker = None
        self.has_belote = False
        self.self_current_score = 0
        self.opponent_current_score = 0

    def bid(self, obs, valid_bids, suits_order):
        raise NotImplementedError()

    def add_cards(self, cards):
        self.cards += cards

    def has_card(self, card):
        return card in self.cards

    def has_suit(self, suit):
        for card in self.cards:
            if card.suit == suit:
                return True
        return False

    def get_suit_card(self, suit):
        suit_cards = []

        for card in self.cards:
            if card.suit == suit:
                suit_cards.append(card)
        return suit_cards

    def remove_card(self, card):
        self.cards.remove(card)

    def play_turn(self, trick, obs, suits_order):
        cards_order = self.get_cards_order(obs, suits_order)
        for card in cards_order:
            try:
                trick.add_card(card, self)
                self.remove_card(card)
                break
            except PlayException as e:
                continue

    def get_cards_order(self, obs, suits_order):
        raise NotImplementedError()


class RandomPlayer(Player):
    def get_cards_order(self, obs, suits_order):
        return sample(self.cards, len(self.cards))

    def bid(self, obs, valid_bids, suits_order):
        return random.choice(valid_bids)

class DeterministicPlayer(RandomPlayer):
    def get_cards_order(self, obs, suits_order):
        return self.cards

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

        # Detect if opening or answering
        partner_index = (self.index + 2) % 4
        partner_bids = [decode_bid_action(bid) for i, bid in enumerate(obs["bids"]) if i % 4 == partner_index]
        if not partner_bids:
            is_opening = True

        if is_opening:
            for suit in suits_order:
                j = has_jack.get(suit, False)
                n = has_nine.get(suit, False)
                count = suit_counts[suit]
                ace_else = any(s != suit and has_ace.get(s, False) for s in suits_order)
                if j and n and (count >= 1 or ace_else):
                    bid_action = encode_bid_action(90, suit)
                elif (j or n) and (count >= 2 or (count >= 1 and ace_else)):
                    bid_action = encode_bid_action(80, suit)

        else:
            if partner_bids:
                partner_bid = partner_bids[-1]
                value, suit = partner_bid
                value_add = 0
                if has_nine.get(suit, False):
                    value_add += 10
                if has_jack.get(suit, False):
                    value_add += 20
                for s in suits_order:
                    if s != suit and has_ace.get(s, False):
                        value_add += 10

                new_value = value + value_add
                if new_value > value:
                    bid_action = encode_bid_action(new_value, suit)
        
        return bid_action if bid_action in valid_bids else 0


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

        # Instantiate shared + two heads
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

    def get_cards_order(self, obs, suits_order):
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
        card_mask = torch.from_numpy(obs["player_cards"]).to(self.device)
        masked = probs * card_mask

        if masked.sum() > 0:
            dist = masked / masked.sum()
            idx = int(torch.multinomial(dist, 1).item())
        else:
            # fallback to random card you hold
            hold_idx = np.where(obs["player_cards"]>0)[0]
            idx = int(random.choice(hold_idx))

        # now build play order: sample first, then the rest in any order
        chosen = Card.from_index(idx, suits_order)
        rest   = [c for c in self.cards if c != chosen]
        return [chosen] + rest
    