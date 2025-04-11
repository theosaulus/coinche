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
    - Action numbers 37...40: capot 
    - 41 is coinche (ie. double the bet, but not the bet value)
    - 42 is surcoinche (ie. double the bet again, but not the bet value)
    - 43 is used for padding, and should not be sampled as an action.

    :param action: action number
    :return: (bid_value, atout_suit) or "pass"
    """
    if action == 0:
        return "pass"
    elif action in range(1, 37):
        bid_value = 80 + ((action - 1) // 4) * 10
        trump_index = (action - 1) % 4
        atout_suit = list(Suit)[trump_index]
        return (bid_value, atout_suit)
    elif action in range(37, 41):
        atout_suit = list(Suit)[trump_index]
        return (250, atout_suit) # capot is encoded as 250
    elif action == 41:
        return "coinche"
    elif action == 42:
        return "surcoinche"
    else:
        raise ValueError("Invalid action: 43 is used for padding and should not be sampled as an action.")

def encode_bid_action(bid_value, suit):
    suit_index = list(Suit).index(suit)
    action = 1 + 4 * ((bid_value - 80) // 10) + suit_index
    return action

def decode_observation(observation, bidding_history_length, suits_order):
    # Takes np.array of observation and returns a dict
    bidding_history = observation[:bidding_history_length]
    played_cards_vector = observation[bidding_history_length:bidding_history_length + 32]
    player_cards_vector = observation[bidding_history_length + 32:bidding_history_length + 64]
    trick_cards_vector = observation[bidding_history_length + 64:bidding_history_length + 96]
    contract_value = int(observation[bidding_history_length + 96])
    current_player_attacker = int(observation[bidding_history_length + 97])

    played_cards = convert_index_to_cards(np.where(played_cards_vector == 1)[0], suits_order)
    player_cards = convert_index_to_cards(np.where(player_cards_vector == 1)[0], suits_order)
    trick_cards = convert_index_to_cards(np.where(trick_cards_vector == 1)[0], suits_order)

    return {
        "bidding_history": bidding_history,
        "played_cards": played_cards,
        "player_cards": player_cards,
        "trick_cards": trick_cards,
        "contract_value": contract_value,
        "current_player_attacker": current_player_attacker,
    }