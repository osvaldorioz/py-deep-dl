#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>

namespace py = pybind11;

class DeepQLearningAgent {
private:
    Eigen::MatrixXd Q_table;
    double alpha;
    double gamma;
    int state_dim;
    int action_dim;
    
public:
    DeepQLearningAgent(int state_dim, int action_dim, double alpha, double gamma)
        : state_dim(state_dim), action_dim(action_dim), alpha(alpha), gamma(gamma) {
        Q_table = Eigen::MatrixXd::Zero(state_dim, action_dim);
    }

    int select_action(const Eigen::VectorXd& state, double epsilon) {
        if (((double)rand() / RAND_MAX) < epsilon) {
            return rand() % action_dim;
        }
        Eigen::VectorXd q_values = Q_table.transpose() * state;
        Eigen::Index max_index;
        q_values.maxCoeff(&max_index);
        return static_cast<int>(max_index);
    }

    void update(const Eigen::VectorXd& state, int action, double reward, const Eigen::VectorXd& next_state) {
        Eigen::VectorXd next_q_values = Q_table.transpose() * next_state;
        double max_next_q = next_q_values.maxCoeff();
        double target = reward + gamma * max_next_q;
        Q_table.col(action) += alpha * (target - Q_table.col(action).dot(state)) * state;
    }
};

PYBIND11_MODULE(deep_rl, m) {
    py::class_<DeepQLearningAgent>(m, "DeepQLearningAgent")
        .def(py::init<int, int, double, double>())
        .def("select_action", &DeepQLearningAgent::select_action)
        .def("update", &DeepQLearningAgent::update);
}
