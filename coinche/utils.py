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
    There are 44 actions: 
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
        trump_index = (action - 1) % 4
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

def obs_dict_to_vector(obs_dict):
    return np.concatenate([
            obs_dict["bids"], 
            obs_dict["played_cards"], 
            obs_dict["player_cards"],
            obs_dict["trick_cards"], 
            obs_dict["current_scores"]/162.0, 
            obs_dict["extra"]
        ], axis=0).astype(np.float32)

def decode_observation(obs_dict, suits_order):
    return {
        "bids": obs_dict["bids"].astype(int).tolist(),
        "played_cards": convert_index_to_cards(np.where(obs_dict["played_cards"]==1)[0], suits_order),
        "player_cards": convert_index_to_cards(np.where(obs_dict["player_cards"]==1)[0], suits_order),
        "trick_cards": convert_index_to_cards(np.where(obs_dict["trick_cards"]==1)[0], suits_order),
        "current_scores": obs_dict["current_scores"].tolist(),
        "extra": obs_dict["extra"].tolist(),
    }
