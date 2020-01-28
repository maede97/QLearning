#ifndef Q_LEARNING_HPP
#define Q_LEARNING_HPP

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <functional>

/**
 * @brief Contains all Q-Learning classes
 * 
 */
namespace QL
{


/**
 * @brief Function type for the transition function
 * 
 */
#define TRANSITION_FUNC_TYPE std::function<QL::BaseState *(QL::BaseState *, QL::BaseAction *, size_t, QL::BaseState *)>

/**
 * @brief Tuple for return value of QL::QLeaner::iterate function
 * 
 */
#define ITERATE_TYPE std::tuple<QL::BaseState *, QL::BaseAction *, QL::BaseState *>

/**
 * @brief Base class for representing a state
 * 
 */
class BaseState
{
};

/**
 * @brief Base class for representing an action
 * 
 */
class BaseAction
{
};

/**
 * @brief Base class for representing a reward
 * 
 */
class BaseReward
{
public:
    /**
     * @brief Reward function, will be evaulauted
     * 
     * @param from_state The old state
     * @param action The action taken
     * @param to_state The new state
     * @return double reward from old state to new one with a given action
     */
    virtual double r(BaseState *from_state, BaseAction *action, BaseState *to_state) = 0;
};

/**
 * @brief Class which represents a Q-Learner
 * 
 */
class QLearner
{
public:
    /**
     * @brief Construct a new QLearner object
     * 
     * @param nstates Number of states
     * @param states Array of all states
     * @param nactions Number of actions
     * @param actions Array of actions, if has higher priority given a state, it must be earlier inside the array
     * @param reward Reward class
     */
    QLearner(size_t nstates, BaseState *states, size_t nactions, BaseAction *actions, BaseReward *reward);

    /**
     * @brief Set up all rates
     * 
     * @param gamma discount factor
     * @param alpha learning rate
     */
    void setRates(double gamma, double alpha);

    /**
     * @brief Iterate the Q-Learning algorithm one step
     * 
     * @return std::tuple<QL::BaseState *, QL::BaseAction *, QL::BaseState *> A tuple, from where with an action to the next state
     */
    ITERATE_TYPE iterate();

    /**
     * @brief Set functor for transition between states given an action
     * 
     * @param f functor
     */
    void setTransitions(TRANSITION_FUNC_TYPE f);

    /**
     * @brief Initizalize Q-Learning with Q_max
     * 
     * @param Q_max value for each Q
     */
    void initWithQMax(double Q_max);
    /**
     * @brief Set the Init State And init action
     * 
     * @param state Initial state
     * @param action initial action
     */
    void setInitStateAndAction(BaseState *state, BaseAction *action);

    /**
     * @brief Get the Current Q values
     * 
     * @return Eigen::MatrixXd containing all Q values (row: state, column: action), orderes as arrays given
     */
    Eigen::MatrixXd getCurrentQ() const;

private:
    /**
     * @brief Get the maximum Q and it's action
     * 
     * @param new_state the state from which we want to get the best Q
     * @return std::pair<BaseAction *, double> action leading needed after new_state and the q value
     */
    std::pair<BaseAction *, double> getMaxQ(BaseState *new_state);

    double gamma_;                             ///< internal gamma
    double alpha_;                             ///< internal alpha
    size_t nstates_;                           ///< internal number of states
    size_t nactions_;                          ///< internal number of actions
    BaseState *states_;                        ///< internal array of states
    BaseState *curr_state_;                    ///< current state of the player
    BaseAction *curr_action_;                  ///< current action of the player (only init)
    BaseAction *actions_;                      ///< array of all actions
    BaseReward *reward_;                       ///< internal reward function
    Eigen::MatrixXd Q_values_;                 ///< internal Q values
    Eigen::MatrixXd Q_values_old_;             ///< internal old Q values
    TRANSITION_FUNC_TYPE transition_function_; ///< internal transition function between states with an action
};

} // namespace QL

#endif