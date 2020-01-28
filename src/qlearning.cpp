#include <qlearning.hpp>

QL::QLearner::QLearner(size_t nstates, BaseState *states, size_t nactions, BaseAction *actions, BaseReward *reward)
    : nstates_(nstates), states_(states), nactions_(nactions), actions_(actions), reward_(reward)
{
    Q_values_ = Eigen::MatrixXd::Zero(nstates, nactions);
    Q_values_old_ = Q_values_;
}

void QL::QLearner::setRates(double gamma, double alpha)
{
    gamma_ = gamma;
    alpha_ = alpha;
}

void QL::QLearner::initWithQMax(double Q_max)
{
    Q_values_.setConstant(Q_max);
    Q_values_old_ = Q_values_;
}

void QL::QLearner::setInitStateAndAction(BaseState *state, BaseAction *action)
{
    curr_state_ = state;
    curr_action_ = action;
}

Eigen::MatrixXd QL::QLearner::getCurrentQ() const
{
    return Q_values_;
}

std::tuple<QL::BaseState *, QL::BaseAction *, QL::BaseState *> QL::QLearner::iterate()
{
    // choose best action from current state, if the same, choose first
    std::pair<QL::BaseAction *, double> bestAction;
    if (curr_action_ == nullptr)
    {
        bestAction = getMaxQ(curr_state_);
    }
    else
    {
        // do initial movement with a given action
        bestAction = {curr_action_, Q_values_((curr_state_ - states_) / sizeof(curr_state_), (curr_action_ - actions_) / sizeof(curr_action_))};
        curr_action_ = nullptr; // nullify
    }

    // do the move
    BaseState *old_state = curr_state_;
    curr_state_ = transition_function_(curr_state_, bestAction.first, nstates_, states_);

    // calculate new offsets
    size_t action_offset = (bestAction.first - actions_) / sizeof(bestAction.first);
    size_t state_offset = (curr_state_ - states_) / (sizeof(curr_state_));

    // update Q_values
    for (size_t a = 0; a < nactions_; a++)
    {
        for (size_t s = 0; s < nstates_; s++)
        {
            QL::BaseState *fromState = states_ + s * sizeof(curr_state_);
            QL::BaseState *toState = transition_function_(fromState, actions_ + a * sizeof(&actions_[0]), nstates_, states_);
            Q_values_(s, a) = (1.0 - alpha_) * Q_values_old_(s, a) + alpha_ * (reward_->r(fromState, actions_ + a * sizeof(&actions_[0]), toState) + gamma_ * bestAction.second);
        }
    }

    // get max Q (greedy)
    size_t row = (old_state - states_) / sizeof(old_state);
    size_t col;
    double maxQ = Q_values_.row(row).maxCoeff(&col);

    // update old values
    Q_values_old_(row, col) = Q_values_(row, col);
    QL::BaseAction *bestAction_A = actions_ + sizeof(&actions_[0]) * col;
    curr_state_ = transition_function_(old_state, bestAction_A, nstates_, states_);

    return {old_state, bestAction_A, curr_state_};
}

void QL::QLearner::setTransitions(TRANSITION_FUNC_TYPE f)
{
    transition_function_ = f;
}

std::pair<QL::BaseAction *, double> QL::QLearner::getMaxQ(BaseState *new_state)
{
    size_t offset = (new_state - states_) / sizeof(new_state); // get position of new state respective to all states
    Eigen::RowVector2d Q_row = Q_values_old_.row(offset);      // get from old ones
    size_t maxQ_i;
    double maxQ = Q_row.maxCoeff(&maxQ_i);
    return {actions_ + maxQ_i * sizeof(&actions_[0]), maxQ};
}