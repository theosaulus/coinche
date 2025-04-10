import numpy as np
import random
import gymnasium as gym

from coinche.player import RandomPlayer, AIPlayer
from coinche.gym.player import GymPlayer
from coinche.trick import Trick
from coinche.deck import Deck
from coinche.card import Suit
from coinche.utils import convert_cards_to_vector
from coinche.reward_prediction import decision_process

from gymnasium import Env, spaces


def make_env(env_id="coinche-v3", seed=None):
    def _init():
        env =  gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init
# to be used like envs = AsyncVectorEnv([make_env(seed=i) for i in range(8)])

def decode_bid_action(action):
    """
    Decode the action number into a bid value and trump suit.
    There are 37 actions: 0 is "pass", 1-36 map to bids.
    Action numbers 1...36: bid_value increases in increments of 10 starting at 80.
    E.g., 1 = 80 heart, 2 = 80 spades, etc.

    :param action: action number
    :return: (bid_value, atout_suit) or "pass"
    """
    if action == 0:
        return "pass"
    bid_value = 80 + ((action - 1) // 4) * 10
    trump_index = (action - 1) % 4
    atout_suit = list(Suit)[trump_index]
    return (bid_value, atout_suit)

class GymCoinche(Env):
    def __init__(self, players=None, contrat_model_path=None):
        # observation_space
        # 32 played cards + 32 player cards + 32 cards of current trick + contract_value + attacker
        # 8 atouts + 8 suit 1 + 8 suit 2 + 8 suit 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(98,))
        # 32 cards
        # 8 atouts + 8 suit 1 + 8 suit 2 + 8 suit 3
        self.action_space = spaces.Discrete(32)

        self.players = players if players is not None else [
            RandomPlayer(0, "N"),
            RandomPlayer(1, "E"),
            GymPlayer(2, "S"),
            RandomPlayer(3, "W")
        ]

        # Theo: Reorganization of the init would be nice to better separate phases
        self.current_trick_rotation = []
        self.deck = Deck()
        self.round_number = 0
        self.played_tricks = []
        self.trick = None
        self.atout_suit = None
        self.contract_value = None
        self.suits_order = None
        self.contrat_model = None # models.load_model(contrat_model_path) if contrat_model_path is not None else None
        print("Contrat model passed: ", contrat_model_path)
        self.attacker_team = 0
        self.original_hands = {}

        self.dealer_index = 0
        self.bidding_history = []
        self.bidding_history_length = 30 # max number of bids possible: 4 + 3*8 + 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)        
        """
        reset is mandatory to use gym framework. Reset is called at the end of each round (8 tricks)
        :return: observation
        """
        # New round
        self.round_number += 1

        # We rebuild the deck based on previous trick won by each players
        self._rebuild_deck(self.played_tricks)
        self._deal_cards()
        self._bidding_phase()
        self.played_tricks = []

        self.original_hands = {
            "player0-hand": convert_cards_to_vector(self.players[0].cards, self.suits_order),
            "player1-hand": convert_cards_to_vector(self.players[1].cards, self.suits_order),
            "player2-hand": convert_cards_to_vector(self.players[2].cards, self.suits_order),
            "player3-hand": convert_cards_to_vector(self.players[3].cards, self.suits_order),
            "attacker_team": [p.attacker for p in self.players]
        }
        self.total_score = 0
        self.trick = Trick(self.atout_suit, trick_number=1)
        self.current_trick_rotation = self._create_trick_rotation(self.round_number % 4)

        # Play until AI
        self._play_until_end_of_rotation_or_ai_play()
        observation = self._get_trick_observation()
        return observation, {}

    def step(self, action):
        """
        step is mandatory to use gym framework
        In each step, every player play exactly one, especially the AIPlayers.
        :param action:
        :return: observation, reward, done, info
        """
        # Play for gym player
        ai_player = self.current_trick_rotation[0]

        action_vector = np.zeros(32)
        action_vector[action] = 1
        ai_player.set_next_action(action_vector)

        ai_player.play_turn(self.trick, self.played_tricks, self.suits_order, self.contract_value)
        self.current_trick_rotation.pop(0)
        # Then play until end of trick
        self._play_until_end_of_rotation_or_ai_play()
        # Handle end of trick
        winner = self.trick.winner
        trick_score_factor = ai_player.index % 2 == winner.index % 2
        reward = self._get_reward(self.trick,
                                  self.total_score,
                                  trick_score_factor,
                                  self.contract_value)
        # add score to teams
        self.played_tricks.append(self.trick)

        # Add trick score to total_score
        self.total_score += self._get_trick_reward(self.trick, trick_score_factor)

        if len(self.played_tricks) < 8:
            self.trick = Trick(self.atout_suit, trick_number=len(self.played_tricks) + 1)
            # Choose next starter
            self.current_trick_rotation = self._create_trick_rotation(winner.index)
            # Play until AI
            self._play_until_end_of_rotation_or_ai_play()
            observation = self._get_trick_observation()
            winning_team = 0 if winner.index % 2 == 0 else 1
            info = {'winner': winner.index,
                    'winning_team': winning_team}
            terminated = False
            return observation, reward, terminated, False, info
        else:
            observation = self._get_round_observation()
            info = self.original_hands
            info["total_reward"] = self.total_score
            terminated = True
            return observation, reward, terminated, False, info

    def _bidding_phase(self):
        """
        Implements the bidding phase using 37 possible actions:
        - 0: pass
        - 1..36: bid (80-160) with suit
        """
        self.bidding_history = []
        current_bid = None
        winning_player = None

        # 1st player is the one on the left of the dealer
        player_index = (self.dealer_index + 1) % 4 
        passes_in_row = 0

        while True:
            player = self.players[player_index]
            valid_bids = self._get_valid_bid_actions(current_bid)
            suits_order = list(Suit)
            hand = convert_cards_to_vector(player.cards, suits_order)

            action = player.bid(hand, self.bidding_history, valid_bids, suits_order)
            if action not in valid_bids:
                raise RuntimeError(f"Invalid action {action} for player {player.index} with hand {hand}")

            self.bidding_history.append((player.index, action))
            bid = decode_bid_action(action)

            if bid == "pass":
                passes_in_row += 1
            else:
                bid_value, bid_suit = bid
                if current_bid is None or bid_value > current_bid[0]:
                    current_bid = (bid_value, bid_suit)
                    winning_player = player
                    passes_in_row = 0
                else:
                    raise RuntimeError(f"Invalid bid {action} for player {player.index} with hand {hand}")

            if current_bid is None and len(self.bidding_history) >= 4:
                # everyone passed
                # Theo: we should allow everyone to pass, but give bad reward to everyone
                current_bid = (80, random.choice(list(Suit)))
                winning_player = self.players[(self.round_number + 1) % 4]
                break

            if current_bid is not None and passes_in_row >= 3:
                # everyone passed after a bid: end of bidding phase
                break

            player_index = (player_index + 1) % 4

        self.contract_value, self.atout_suit = current_bid
        self.attacker_team = winning_player.index % 2
        self.suits_order = Suit.create_order(self.atout_suit)

        for p in self.players:
            p.attacker = int(p.index % 2 == self.attacker_team)
    
    def _set_contrat(self, contrat_model):
        default_suit_order = list(Suit)

        hand0 = convert_cards_to_vector(self.players[0].cards, default_suit_order)
        hand1 = convert_cards_to_vector(self.players[1].cards, default_suit_order)
        hand2 = convert_cards_to_vector(self.players[2].cards, default_suit_order)
        hand3 = convert_cards_to_vector(self.players[3].cards, default_suit_order)

        shift_team1, expected_reward_team1 = decision_process(hand0, hand2, contrat_model)
        shift_team2, expected_reward_team2 = decision_process(hand1, hand3, contrat_model)

        if expected_reward_team1 > expected_reward_team2:
            expected_reward_team = expected_reward_team1
            attacker_team = 0
            shift = shift_team1
        else:
            expected_reward_team = expected_reward_team2
            attacker_team = 1
            shift = shift_team2

        # Carefull: shifting is anti-clockwise
        self.atout_suit = default_suit_order[-shift]
        self.suits_order = Suit.create_order(self.atout_suit)

        self.contract_value = np.max([0, (expected_reward_team//10) - 8])/9
        self.attacker_team = attacker_team

    def _rebuild_deck(self, played_tricks):
        # Check if duplicated cards
        for p in self.players:
            for trick in played_tricks:
                if p.index == trick.winner.index:
                    self.deck.add_trick(trick)
        self.deck.cut_deck()

    def _deal_cards(self):
        """
        This function deals the deck
        """
        players_round = np.roll(self.players, self.round_number)
        legal_dealing_sequences = [[3, 3, 2], [3, 2, 3]]  # Defining academic dealing sequences
        dealing_sequence = legal_dealing_sequences[random.randint(0, 1)]  # Choose the Dealing Sequence
        for cards_to_deal in dealing_sequence:
            for p in players_round:  # Stopping condition on one round
                p.add_cards(self.deck.deal(cards_to_deal))

    def _play_until_end_of_rotation_or_ai_play(self):
        """
        When this method is called it is either:
        - An AIPlayer's turn: therefore we break and ask the AIPlayer to give prediction
        - Not an AIPlayer: therefore the current player just play given is deterministic (or random) policy
        :return:
        """
        while len(self.current_trick_rotation) > 0:
            current_player = self.current_trick_rotation[0]
            if isinstance(current_player, GymPlayer):
                break
            current_player.play_turn(self.trick, self.played_tricks, self.suits_order, self.contract_value)
            self.current_trick_rotation.pop(0)

    def _get_trick_observation(self):
        # self.observation_space = [spaces.Discrete(2)] * (32 + 32 + 32) + [spaces.Discrete(10), spaces.Discrete(2)]
        played_cards = [card for trick in self.played_tricks for card in trick.cards]
        played_cards_observation = convert_cards_to_vector(played_cards, self.suits_order)
        current_player = self.current_trick_rotation[0]
        player_cards_observation = convert_cards_to_vector(current_player.cards, self.suits_order)
        trick_cards_observation = convert_cards_to_vector(self.trick.cards, self.suits_order)
        observation = np.concatenate((played_cards_observation,
                                      player_cards_observation,
                                      trick_cards_observation,
                                      [self.contract_value, current_player.attacker]))
        return observation.astype(np.float32)

    def _get_round_observation(self):
        # self.observation_space = [spaces.Discrete(2)] * (32 + 32 + 32) + [spaces.Discrete(10), spaces.Discrete(2)]
        played_cards_observation = np.ones(32)
        player_cards_observation = np.zeros(32)
        trick_cards_observation = convert_cards_to_vector(self.trick.cards, self.suits_order)
        observation = np.concatenate((played_cards_observation,
                                      player_cards_observation,
                                      trick_cards_observation,
                                      [self.contract_value, 1]))
        return observation.astype(np.float32)

    def _get_valid_bid_actions(self, current_bid):
        if current_bid is None:
            return list(range(1, 37)) + [0]  # all possible bids + pass
        else:
            min_bid_value = current_bid[0] + 10
            valid = []
            for i in range(1, 37):
                val, suit = decode_bid_action(i)
                if val >= min_bid_value:
                    valid.append(i)
            valid.append(0)  # pass always allowed
            return valid

    def _create_trick_rotation(self, starting_player_index):
        rotation = np.array(self.players)
        while rotation[0].index != starting_player_index:
            rotation = np.roll(rotation, 1)
        return rotation.tolist()

    def _get_trick_reward(self, trick, trick_score_factor):
        score = trick.score()
        return score * trick_score_factor

    def _get_reward(self, trick, total_score, trick_score_factor, value, normalisation_trick=10):
        # trick reward: if not last trick
        if True:
            score = trick.score()
            if trick_score_factor:
                return score  / normalisation_trick
            else:
                return - score / normalisation_trick
        # if True:
        #     score = trick.score()
        #     if trick_score_factor:
        #         return np.exp(score  / normalisation_trick)
        #     else:
        #         return - np.exp(score / normalisation_trick)

        # if this is last trick of round
#         else:
# #          Let's check if contract is done
#             if ((total_score/10) - 8)/9 >= value:
#                 return 9
#             else:
#                 return -9

