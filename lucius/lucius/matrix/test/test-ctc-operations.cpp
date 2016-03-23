#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include <iostream>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CTCOperations.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>

//CTC Includes
#include <lucius/matrix/ctc/interface/ctc.h>
#include <lucius/matrix/ctc/tests/test.h>

namespace lucius
{
namespace matrix
{
namespace ctc
{
//void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations, const Matrix& reference);

bool small_test() {
//  std::vector<float> activations = {0.1, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1};
    const int alphabetSize = 5;
    const int timeSteps    = 2;
    Matrix inputActivations(alphabetSize, 1, timeSteps);
    
    inputActivations(0,0,0) = 0.1;
    inputActivations(1,0,0) = 0.6;
    inputActivations(2,0,0) = 0.1;
    inputActivations(3,0,0) = 0.1;
    inputActivations(4,0,0) = 0.1;
    inputActivations(0,0,1) = 0.1;
    inputActivations(1,0,1) = 0.1;
    inputActivations(2,0,1) = 0.6;
    inputActivations(3,0,1) = 0.1;
    inputActivations(4,0,1) = 0.1;

    // Calculate the score analytically
    Matrix probs = softmax(inputActivations);

    // Score calculation is specific to the given activations above
    float expectedScore = probs(1, 0, 0) * probs(2, 0, 1);
    
    Matrix referenceCosts({1});
    
    referenceCosts(0) = expectedScore;
    
    Matrix costs({1});
    Matrix gradients(inputActivations.size());
    
    Matrix referenceLabels = zeros(inputActivations.size(), inputActivations.precision());
    
    referenceLabels(1, 0, 0) = 1.0;
    referenceLabels(2, 0, 1) = 1.0;
    std::cout << "inputActivations: " << inputActivations.toString() << "\n";
    std::cout << "referenceLabels: " << referenceLabels.toString() << std::endl;

    computeCtc(costs, gradients, inputActivations, referenceLabels);
    std::cout << "costs: " << costs.toString();

    float score = costs(0);
    std::cout << "Score: " << score << "\n";
    std::cout << "Gradients: " << gradients.toString();
    score = std::exp(-score);
    const float eps = 1e-6;

    const float lb = expectedScore - eps;
    const float ub = expectedScore + eps;

    return (score > lb && score < ub);
}

bool inf_test() {
    const int alphabet_size = 15;
    const int T = 50;
    const int L = 10;
    const int minibatch = 1;

    std::vector<int> labels = genLabels(alphabet_size, L);
    labels[0] = 2;
    std::vector<int> label_lengths = {L};

    std::vector<float> acts = genActs(alphabet_size * T * minibatch);

    for (int i = 0; i < T; ++i)
        acts[alphabet_size * i + 2] = -1e30;

    std::vector<int> sizes;
    sizes.push_back(T);

    std::vector<float> grads(alphabet_size * T);

    float cost;

    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), sizes.data(),
                                      alphabet_size, sizes.size(), info,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in inf_test");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_ctc_loss(acts.data(), grads.data(),
                                    labels.data(), label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    sizes.size(),
                                    &cost,
                                    ctc_cpu_workspace,
                                    info),
                   "Error: compute_ctc_loss in inf_test");

    free(ctc_cpu_workspace);

    bool status = true;
    status &= std::isinf(cost);

    for (int i = 0; i < alphabet_size * T; ++i)
        status &= !std::isnan(grads[i]);
 
    return status;
}

float grad_check(int T, int alphabet_size,
                  std::vector<float>& acts,
                  const std::vector<std::vector<int>>& labels,
                  const std::vector<int>& sizes) {

    float epsilon = 1e-2;

    const int minibatch = labels.size();

    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (const auto& l : labels) {
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }

    std::vector<float> costs(minibatch);

    std::vector<float> grads(acts.size());

    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(), sizes.data(),
                                      alphabet_size, sizes.size(), info,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in grad_check");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_ctc_loss(acts.data(), grads.data(),
                                    flat_labels.data(), label_lengths.data(),
                                    sizes.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_cpu_workspace,
                                    info),
                   "Error: compute_ctc_loss (0) in grad_check");

    std::vector<float> num_grad(grads.size());

    //perform 2nd order central differencing
    for (int i = 0; i < T * alphabet_size * minibatch; ++i) {

        std::vector<float> costsP1(minibatch);
        std::vector<float> costsP2(minibatch);

        acts[i] += epsilon;
        throw_on_error(compute_ctc_loss(acts.data(), NULL,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP1.data(),
                                        ctc_cpu_workspace,
                                        info),
                       "Error: compute_ctc_loss (1) in grad_check");

        acts[i] -= 2 * epsilon;
        throw_on_error(compute_ctc_loss(acts.data(), NULL,
                                        flat_labels.data(), label_lengths.data(),
                                        sizes.data(),
                                        alphabet_size,
                                        minibatch,
                                        costsP2.data(),
                                        ctc_cpu_workspace,
                                        info),
                       "Error: compute_ctc_loss (2) in grad_check");

        float costP1 = std::accumulate(costsP1.begin(), costsP1.end(), 0.);
        float costP2 = std::accumulate(costsP2.begin(), costsP2.end(), 0.);

        acts[i] += epsilon;
        num_grad[i] = (costP1 - costP2) / (2 * epsilon);
    }

    free(ctc_cpu_workspace);

    float diff = rel_diff(grads, num_grad);

    return diff;
}

bool run_tests() {
    std::vector<std::tuple<int, int, int, int, float>> problem_sizes =
       {std::make_tuple(20, 50, 15, 1, 1e-5),
        std::make_tuple(5, 10, 5, 65, 1e-4)
       };

    std::mt19937 gen(2);

    bool status = true;
    for (auto problem : problem_sizes) {
        int alphabet_size, T, L, minibatch;
        float tol;
        std::tie(alphabet_size, T, L, minibatch, tol) = problem;

        std::vector<float> acts = genActs(alphabet_size * T * minibatch);

        std::vector<std::vector<int>> labels;
        std::vector<int> sizes;
        for (int mb = 0; mb < minibatch; ++mb) {
            int actual_length = L;
            labels.push_back(genLabels(alphabet_size, actual_length));
            sizes.push_back(T);
        }

        float diff = grad_check(T, alphabet_size, acts, labels, sizes);

        status &= (diff < tol);
    }

    return status;
}
}
}
}

int main(void) {
    std::cout << "Running CTC operations tests" << std::endl;

    bool status = true;
    status &= lucius::matrix::ctc::small_test();
    //status &= lucius::matrix::ctc::inf_test();
    //status &= lucius::matrix::ctc::run_tests();

    if (status)
        std::cout << "Tests pass" << std::endl;
    else
        std::cout << "Some or all tests fail" << std::endl;
}
