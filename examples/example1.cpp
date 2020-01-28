#include <qlearning.hpp>
#include <iostream>

class MyState : public QL::BaseState
{
public:
    MyState() {}
    MyState(size_t x) : x_(x)
    {
    }

    size_t getX() const { return x_; }

private:
    size_t x_;
};

#define LEFT 0
#define RIGHT 1
class MyAction : public QL::BaseAction
{
public:
    MyAction() {}
    MyAction(size_t i) : i_(i) {}

    size_t getI() const { return i_; }

private:
    size_t i_;
};

class MyReward : public QL::BaseReward
// reward only dependent on previous state before action
{
public:
    double r(QL::BaseState *from_state, QL::BaseAction *action, QL::BaseState *to_state)
    {
        MyState *mystate = reinterpret_cast<MyState *>(from_state);
        MyAction *myaction = reinterpret_cast<MyAction *>(action);

        if (mystate->getX() == 1)
        {
            return 20;
        }
        else
        {
            return 30;
        }
    }
};

QL::BaseState *transition(QL::BaseState *fromQL, QL::BaseAction *actionQL, size_t nstates, QL::BaseState *statesQL)
{
    MyState *from = reinterpret_cast<MyState *>(fromQL);
    MyAction *action = reinterpret_cast<MyAction *>(actionQL);

    if (from->getX() == 1 && action->getI() == RIGHT)
    {
        return from + 1;
    }
    else if (from->getX() == 2 && action->getI() == LEFT)
    {
        return statesQL;
    }
    else
    {
        return from;
    }
}

int main(int argc, char const *argv[])
{
    MyReward reward;
    MyState states[2];
    states[0] = MyState(1);
    states[1] = MyState(2);

    MyAction actions[2];
    actions[0] = MyAction(RIGHT);
    actions[1] = MyAction(LEFT);

    // set up learner
    QL::QLearner learner(2, states, 2, actions, &reward);
    learner.initWithQMax(100.0);
    learner.setInitStateAndAction(&states[0], &actions[0]);
    learner.setTransitions(transition);
    learner.setRates(0.5, 0.5);

    ITERATE_TYPE newPos;

    for (int i = 0; i < 5; i++)
    {
        newPos = learner.iterate();
        MyState *oldState = reinterpret_cast<MyState *>(std::get<0>(newPos));
        MyState *newState = reinterpret_cast<MyState *>(std::get<2>(newPos));
        MyAction *newAction = reinterpret_cast<MyAction *>(std::get<1>(newPos));

        std::cout << oldState->getX() << ((newAction->getI() == LEFT) ? " left " : " right ") << newState->getX() << std::endl;
        std::cout << learner.getCurrentQ() << std::endl;
    }

    return 0;
}
