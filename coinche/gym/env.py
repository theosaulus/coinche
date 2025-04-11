import numpy as np
import random
import gymnasium as gym

from coinche.player import RandomPlayer, AIPlayer
from coinche.gym.gymplayer import GymPlayer
from coinche.trick import Trick
from coinche.deck import Deck
from coinche.card import Suit, Card
from coinche.utils import convert_cards_to_vector, decode_bid_action
from coinche.reward_prediction import decision_process

from gymnasium import Env, spaces

PAD_ACTION = 43

def make_env(env_id="coinche-v3", seed=None):
    def _init():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init
# to be used like envs = AsyncVectorEnv([make_env(seed=i) for i in range(8)])


class GymCoinche(Env):
    def __init__(self, players=None):
        # observation_space
        # 37 bids placed (at most) + 32 played cards + 32 player cards + 32 cards of current trick + attacker + bidding/trick phase
        # TODO: BIDS MUST BE BETTER ENCODED
        self.bidding_history_length = 37 # 4 + 3 * 11, because cardinal(90 to 160 + 250 + coinche + surcoinche)=11
        self.other_state_length = 98
        self.observation_space = spaces.Box(low=0, high=1, shape=(135,))
        # 32 cards: 8 atouts + 8 suit 1 + 8 suit 2 + 8 suit 3
        # 44 bids: pass + (80 to 160 + capot) for each suit + coinche + surcoinche + padding
        self.action_space = spaces.Discrete(44)

        self.players = players if players is not None else [
            RandomPlayer(0, "N"),
            RandomPlayer(1, "E"),
            GymPlayer(2, "S"),
            RandomPlayer(3, "W")
        ]
        self.tricks_reward_factor = 0.1
        self.deck = Deck()
        self.round_number = 0
        self.reshuffle_deck_each_round = True

        self.dealer_index = 0
        self.current_bidding_player_index = (self.dealer_index + 1) % 4
        self.bidding_history = []
        self.current_bid = None # tuple: (bid_value, atout_suit)
        self.bid_winning_player = None
        self.passes_in_row = 0

        self.atout_suit = None
        self.contract_value = None
        self.coinche_surcoinche = 0
        self.bidding_done = False

        self.attacker_team = 0
        self.current_trick_rotation = []
        self.played_tricks = []
        self.trick = None
        self.suits_order = None
        self.original_hands = {}
        self.total_score = 0


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.round_number += 1
        self._rebuild_deck(self.played_tricks)
        self._deal_cards()
        self._init_bidding_phase()
        self.played_tricks = []
        return self._get_current_observation(), {}


    def step(self, action):
        """
        step is mandatory to use gym framework
        In each step, every player play exactly one, especially the AIPlayers.
        :param action: action to play (either bid or trick)
        :return: observation, reward, done, info
        """
        if not self.bidding_done:
            return self.bidding_step(action)
        else:
            return self.trick_step(action)

    def bidding_step(self, action):
        player = self.players[self.current_bidding_player_index]
        if not isinstance(player, GymPlayer):
            raise RuntimeError("Not GymPlayer's turn to bid")

        valid_bids = self._get_valid_bid_actions(self.current_bid)
        if action not in valid_bids:
            raise RuntimeError(f"Invalid bid {action} for player {player.index}")

        self._process_bidding(action, player)
        self.current_bidding_player_index = (self.current_bidding_player_index + 1) % 4

        # Play automatically for other players until GymPlayer or end
        while not isinstance(self.players[self.current_bidding_player_index], GymPlayer) and not self.bidding_done:
            player = self.players[self.current_bidding_player_index]
            valid_bids = self._get_valid_bid_actions(self.current_bid)
            action = player.bid(self.bidding_history, valid_bids, list(Suit))
            
            self._process_bidding(action, player)
            self.current_bidding_player_index = (self.current_bidding_player_index + 1) % 4

        if self.current_bid is None and len(self.bidding_history) >= 4:
            # Everyone passed without bid: end of the round, bad reward to everyone
            observation = self._get_current_observation()
            reward = -10
            info = self.original_hands
            info["total_reward"] = -10
            terminated = True
            return observation, reward, terminated, False, info

        elif self.current_bid is not None and self.passes_in_row >= 3:
            self.bidding_done = True
            self.contract_value, self.atout_suit = self.current_bid
            self.attacker_team = self.bid_winning_player.index % 2
            self.suits_order = Suit.create_order(self.atout_suit)
            self.start_trick_phase()
            info = {}

        observation = self._get_current_observation()
        reward = 0
        terminated = False
        return observation, reward, terminated, False, info

    def start_trick_phase(self):
        for p in self.players:
            p.attacker = int(p.index % 2 == self.attacker_team)
            if p.attacker:
                p.has_belote = p.has_card(Card(5, self.atout_suit)) and p.has_card(Card(6, self.atout_suit))

        self.original_hands = {
            f"player{i}-hand": convert_cards_to_vector(player.cards, self.suits_order)
            for i, player in enumerate(self.players)
        }

        self.original_hands["attacker_team"] = [p.attacker for p in self.players]
        self.total_score = 0
        self.trick = Trick(self.atout_suit, trick_number=1)
        self.current_trick_rotation = self._create_trick_rotation(self.round_number % 4)
        self._play_until_end_of_rotation_or_ai_play() # Play until AI

    def trick_step(self, action):
        ai_player = self.current_trick_rotation[0]
        if not isinstance(ai_player, GymPlayer):
            raise RuntimeError("Not GymPlayer's turn to play")

        action_vector = np.zeros(32)
        action_vector[action] = 1
        ai_player.set_next_action(action_vector)

        ai_player.play_turn(self.trick, self.played_tricks, self.suits_order, self.contract_value)
        self.current_trick_rotation.pop(0)

        # Then play until end of trick
        self._play_until_end_of_rotation_or_ai_play()

        # Handle end of trick
        winner = self.trick.winner
        trick_score_factor = (ai_player.index % 2 == winner.index % 2) * self.tricks_reward_factor
        reward = self._get_trick_reward(self.trick, trick_score_factor)
        self.played_tricks.append(self.trick) # add score to teams

        # Add trick score to total_score
        self.total_score += self._get_trick_reward(self.trick, trick_score_factor=1)

        if len(self.played_tricks) < 8:
            self.trick = Trick(self.atout_suit, trick_number=len(self.played_tricks) + 1)
            # Choose next starter
            self.current_trick_rotation = self._create_trick_rotation(winner.index)
            # Play until AI
            self._play_until_end_of_rotation_or_ai_play()
            observation = self._get_current_observation()
            winning_team = 0 if winner.index % 2 == 0 else 1
            info = {'winner': winner.index,
                    'winning_team': winning_team}
            terminated = False
            return observation, reward, terminated, False, info
        else:
            # Compute points as in https://www.ffbelote.org/belote-contree/, Points annonces
            observation = self._get_round_observation()
            info = self.original_hands

            # # add 20 points in case of belote
            # if self.players[self.bid_winning_player.index].has_belote or self.players[(self.bid_winning_player.index + 2) % 4].has_belote:
            #     self.total_score += 20
            #     info["belote"] = True
            # else:
            #     info["belote"] = False
            # # detect if capot
            # attacker_trick_count = sum(1 for t in self.played_tricks if t.winner.index % 2 == self.attacker_team)
            # if attacker_trick_count == 8:
            #     self.total_score = 250  # overwrite any total score
            #     info["capot"] = True
            # info["total_reward"] = self.total_score
            # terminated = True

            attacker_score = self.total_score
            contract = self.contract_value
            capot_announced = (contract == 250)
            capot_realized = sum(t.winner.index % 2 == self.attacker_team for t in self.played_tricks) == 8

            multiplier = 1
            if self.coinche_surcoinche == 1:
                multiplier = 2
            elif self.coinche_surcoinche == 2:
                multiplier = 4

            attacker_points = 0
            defender_points = 0

            if capot_announced:
                if capot_realized:
                    attacker_points = 250 * multiplier
                else:
                    defender_points = 250 * multiplier
            elif attacker_score >= contract:
                attacker_points = contract * multiplier
            else:
                defender_points = 160 * multiplier

            belote_bonus = 20 if any(p.has_belote for p in self.players if p.attacker) else 0
            attacker_points += belote_bonus

            info["capot_realized"] = capot_realized
            info["capot_announced"] = capot_announced
            info["contract_value"] = contract
            info["contract_realized"] = attacker_score >= contract
            info["belote"] = belote_bonus > 0

            info["attacker_score_raw"] = attacker_score
            info["defender_score_raw"] = 162 - attacker_score

            info["total_attacker_points"] = attacker_points
            info["total_defender_points"] = defender_points

            terminated = True
            reward = 0
            return observation, reward, terminated, False, info


    def _init_bidding_phase(self):
        self.bidding_history = []
        self.current_bid = None
        self.bid_winning_player = None
        self.passes_in_row = 0
        self.current_bidding_player_index = (self.dealer_index + 1) % 4
        self.bidding_done = False

    def _process_bidding(self, action, player):
        self.bidding_history.append(action)
        bid = decode_bid_action(action)

        if bid == "pass":
            self.passes_in_row += 1
        elif bid == "coinche":
            if self.current_bid is None:
                raise RuntimeError("Coinche is not allowed before a bid.")
            self.coinche_surcoinche = 1
            self.bid_winning_player = player
            self.passes_in_row = 0
        elif bid == "surcoinche":
            if self.current_bid is None or self.coinche_surcoinche != 1:
                raise RuntimeError("Surcoinche is not allowed before a coinche.")
            self.coinche_surcoinche = 2
            self.bid_winning_player = player
            self.passes_in_row = 0
        else:
            bid_value, bid_suit = bid
            self.current_bid = (bid_value, bid_suit)
            self.bid_winning_player = player
            self.passes_in_row = 0
    
    def _rebuild_deck(self, played_tricks):
        if self.reshuffle_deck_each_round:
            self.deck = Deck()
            self.deck.shuffle()
        else:
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

    def _get_current_observation(self):
        # self.observation_space = [spaces.Discrete(2)] * (32 + 32 + 32) + [spaces.Discrete(10), spaces.Discrete(2)]
        if not self.bidding_done:
            played_cards = []
            current_player = self.players[self.current_bidding_player_index]
            suits_order = list(Suit)
            current_player_attacker = 2 # 2 = bidding phase
            trick_cards_observation = np.zeros(32) # no cards played yet
        
        else:
            played_cards = [card for trick in self.played_tricks for card in trick.cards]
            current_player = self.current_trick_rotation[0]
            suits_order = self.suits_order
            current_player_attacker = current_player.attacker
            trick_cards_observation = convert_cards_to_vector(self.trick.cards, suits_order)

        bidding_history = [PAD_ACTION] * (self.bidding_history_length - len(self.bidding_history)) + self.bidding_history
        bidding_history_observation = np.array(bidding_history)

        played_cards_observation = convert_cards_to_vector(played_cards, suits_order)
        player_cards_observation = convert_cards_to_vector(current_player.cards, suits_order)

        observation = np.concatenate((bidding_history_observation,
                                      played_cards_observation,
                                      player_cards_observation,
                                      trick_cards_observation,
                                      [self.contract_value, current_player_attacker]))
        return observation.astype(np.float32)

    def _get_round_observation(self):
        # self.observation_space = [spaces.Discrete(2)] * (32 + 32 + 32) + [spaces.Discrete(10), spaces.Discrete(2)]
        bidding_history = [PAD_ACTION] * (self.bidding_history_length - len(self.bidding_history)) + self.bidding_history
        bidding_history_observation = np.array(bidding_history)

        played_cards_observation = np.ones(32)
        player_cards_observation = np.zeros(32)
        trick_cards_observation = convert_cards_to_vector(self.trick.cards, self.suits_order)
        
        observation = np.concatenate((bidding_history_observation,
                                      played_cards_observation,
                                      player_cards_observation,
                                      trick_cards_observation,
                                      [self.contract_value, 1]))
        return observation.astype(np.float32)

    def _get_valid_bid_actions(self, current_bid):
        if current_bid is None:
            return list(range(1, 36)) + [0]  # all bids except coinche/surcoinche + pass
        elif current_bid[0] == "coinche":
            return [38, 0] # surcoinche + pass
        else:
            min_bid_value = current_bid[0] + 10
            min_action_index = 1 + 4 * ((min_bid_value - 80) // 10)
            valid = list(range(min_action_index, 37)) + [0]  # all possible bids except surcoinche + pass
            return valid

    def _create_trick_rotation(self, starting_player_index):
        rotation = np.array(self.players)
        while rotation[0].index != starting_player_index:
            rotation = np.roll(rotation, 1)
        return rotation.tolist()

    def _get_trick_reward(self, trick, trick_score_factor):
        score = trick.score() + 10 * (len(self.played_tricks) == 7) # add 10 to last trick
        return score * trick_score_factor
