#include "cmaes.h"

#include <torch/optim/adam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>

namespace torch {
    namespace optim {

        CMAESOptions::CMAESOptions(double sigma0) : sigma0_(sigma0) {}

        bool operator==(const CMAESOptions& lhs, const CMAESOptions& rhs) {
            return (lhs.sigma0() == rhs.sigma0());
        }

        void CMAESOptions::serialize(torch::serialize::OutputArchive& archive) const {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(sigma0);
        }

        void CMAESOptions::serialize(torch::serialize::InputArchive& archive) {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, sigma0);
        }

        bool operator==(const CMAESParamState& lhs, const CMAESParamState& rhs) {
            return (lhs.step() == rhs.step()) &&
                lhs.sigma() == rhs.sigma() &&
                lhs.max_eval() == rhs.max_eval() &&
                lhs.pop_size() == rhs.pop_size() &&
                lhs.f_target() == rhs.f_target();
        }

        void CMAESParamState::serialize(torch::serialize::OutputArchive& archive) const {
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(sigma);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_eval);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(pop_size);
            _TORCH_OPTIM_SERIALIZE_TORCH_ARG(f_target);
        }

        void CMAESParamState::serialize(torch::serialize::InputArchive& archive) {
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, sigma);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, max_eval);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, pop_size);
            _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, f_target);
        }

        Tensor CMAES::step(LossClosure closure) {

        }
    }
}