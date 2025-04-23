import numpy as np

from coinche.player import Player
from coinche.utils import convert_cards_to_vector, convert_index_to_cards
from coinche.exceptions import PlayException
from coinche.card import Card, Suit
from coinche.trick import Trick

class GymPlayer(Player):
    def __init__(self, *args, **kwargs):
        super(GymPlayer, self).__init__(*args, **kwargs)
        self.next_action = None

    def set_next_action(self, action):
        self.next_action = action

    def play_trick(self, trick, obs, suits_order):
        if self.next_action is None:
            raise RuntimeError("No action set for GymPlayer.play_trick")
        
        action = self.next_action
        self.next_action = None

        action_card = convert_index_to_cards(action, suits_order)[0]
        if trick._assert_valid_play_TrueFalse(action_card, self):
            trick.add_card(action_card, self)
            self.remove_card(action_card)
        else:
            print(f"GymPlayer: invalid action {action} and action card is {action_card}.")
            trick._assert_valid_play(action_card, self) # will raise an exception explaining the error

    def bid(self, bidding_history, valid_bids, suits_order):
        if self.next_action is None:
            raise RuntimeError("GymPlayer bidding: no action set.")
        action = self.next_action
        self.next_action = None
        if action not in valid_bids:
            raise ValueError("Invalid action: action is not in valid bids.")
        return action