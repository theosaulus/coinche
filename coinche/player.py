import numpy as np
import torch
import torch.nn as nn
import random

from random import choice, sample
from coinche.utils import convert_cards_to_vector, convert_index_to_cards, decode_bid_action, encode_bid_action
from coinche.exceptions import PlayException
from coinche.card import Suit

class Player:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.cards = []
        self.attacker = None
        self.has_belote = False

    def bid(self, observation, valid_bids, suits_order):
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

    def play_turn(self, trick, played_tricks, suits_order, contract_value):
        cards_order = self.get_cards_order(trick, played_tricks, suits_order, contract_value)
        for card in cards_order:
            try:
                trick.add_card(card, self)
                self.remove_card(card)
                break
            except PlayException as e:
                continue

    def get_cards_order(self, trick, played_tricks, suits_order, contract_value):
        raise NotImplementedError()


class RandomPlayer(Player):
    def get_cards_order(self, _trick, _played_tricks, _suits_order, _contract_value):
        return sample(self.cards, len(self.cards))

    def bid(self, observation, valid_bids, suits_order):
        return random.choice(valid_bids)

class DeterministicPlayer(RandomPlayer):
    def get_cards_order(self, trick, _played_tricks, _suits_order, _contract_value):
        return self.cards

    def bid(self, observation, valid_bids, suits_order):
        for suit_index, suit in enumerate(suits_order):
            for bid_value in [80, 90]:
                action = 1 + 4 * ((bid_value - 80) // 10) + suit_index
                if action in valid_bids:
                    return action
        return 0

    def bid(self, bidding_history, valid_bids, suits_order):
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
        partner_bids = [decode_bid_action(bid) for i, bid in enumerate(bidding_history) if i % 4 == partner_index]
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

    
class TorchPolicy(nn.Module):
    def __init__(self, input_dim=98, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),  # 32 possible cards
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class AIPlayer(Player):
    def __init__(self, policy_path=None, *args, **kwargs):
        super(AIPlayer, self).__init__(*args, **kwargs)
        self.model = TorchPolicy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if policy_path:
            self.model.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.model.eval()

    def get_cards_order(self, trick, played_tricks, suits_order, contract_value):
        played_cards = [card for trick in played_tricks for card in trick.cards]
        played_cards_obs = convert_cards_to_vector(played_cards, suits_order)
        player_cards_obs = convert_cards_to_vector(self.cards, suits_order)
        trick_cards_obs = convert_cards_to_vector(trick.cards, suits_order)
        obs = np.concatenate((played_cards_obs, player_cards_obs, trick_cards_obs, [contract_value / 9, self.attacker]))

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(obs_tensor).squeeze(0).cpu().numpy()

        masked_logits = logits * player_cards_obs
        if np.max(masked_logits) > 0:
            card_indices = np.argsort(-masked_logits)
        else:
            card_indices = np.argsort(-player_cards_obs)
        return convert_index_to_cards(card_indices, suits_order)
    
    def bid(self, observation, valid_bids, suits_order):
        # Have part of the NN to predict the bid
        # Have some shared weights with the tricks, potentially
        return NotImplementedError()