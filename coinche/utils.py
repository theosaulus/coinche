import numpy as np
from coinche.card import Card, Suit

def convert_cards_to_vector(cards, suits_order):
    cards_vector = np.zeros(32)

    for card in cards:
        card_position = card.to_index(suits_order)
        np.put(cards_vector, card_position, 1)
    return cards_vector


def convert_index_to_cards(cards_index, suits_order):
    return [Card.from_index(card_index, suits_order) for card_index in cards_index]


def decode_bid_action(action):
    """
    Decode the action number into a bid value and trump suit.
    There are 40 actions: 
    - 0 is "pass", 1-36 map to bids.
    - Action numbers 1...36: bid_value increases in increments of 10 starting at 80.
        E.g., 1 = 80 heart, 2 = 80 spades, etc.
    - 37 is coinche (ie. double the bet, but not the bet value)
    - 38 is surcoinche (ie. double the bet again, but not the bet value)
    - 39 is used for padding, and should not be sampled as an action.

    :param action: action number
    :return: (bid_value, atout_suit) or "pass"
    """
    if action == 0:
        return "pass"
    elif action == 37:
        return "coinche"
    elif action == 38:
        return "surcoinche"
    elif action == 39:
        raise ValueError("Invalid action: 39 is used for padding and should not be sampled as an action.")
    else:
        bid_value = 80 + ((action - 1) // 4) * 10
        trump_index = (action - 1) % 4
        atout_suit = list(Suit)[trump_index]
        return (bid_value, atout_suit)
