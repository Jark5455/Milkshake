#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch {
    namespace serialize {
        class OutputArchive;
        class InputArchive;
    }
}

namespace torch {
    namespace optim {
        struct TORCH_API CMAESOptions : public OptimizerCloneableOptions<CMAESOptions> {
            CMAESOptions(double sigma0 = 0.1);

            TORCH_ARG(double, sigma0) = 0.1;
            TORCH_ARG(int64_t, max_eval) = 1000;

            public:
                void serialize(torch::serialize::InputArchive& archive) override;
                void serialize(torch::serialize::OutputArchive& archive) const override;

                TORCH_API friend bool operator==(
                        const CMAESOptions& lhs,
                        const CMAESOptions& rhs);

                double get_lr() const override;
                void set_lr(const double sigma) override;
        };

        struct TORCH_API CMAESParamState : public OptimizerCloneableParamState<CMAESParamState> {
            TORCH_ARG(int64_t, step) = 0;
            TORCH_ARG(torch::Tensor, x);
            TORCH_ARG(torch::Tensor, pop_size);

            public:
                void serialize(torch::serialize::InputArchive& archive) override;
                void serialize(torch::serialize::OutputArchive& archive) const override;

                TORCH_API friend bool operator==(
                        const CMAESParamState& lhs,
                        const CMAESParamState& rhs);
        };

        class TORCH_API CMAES : public Optimizer {
            public:
                explicit CMAES(std::vector<OptimizerParamGroup> param_groups, CMAESOptions defaults = {}) : Optimizer(std::move(param_groups), std::make_unique<CMAESOptions>(defaults))
                {
                    TORCH_CHECK(defaults.sigma0() >= 0, "Invalid initial step size: ", defaults.sigma0());
                    TORCH_CHECK(defaults.max_eval() >= 0, "Invalid max eval: ", defaults.max_eval());
                }

                explicit CMAES(std::vector<Tensor> params, CMAESOptions defaults = {}) : CMAES({OptimizerParamGroup(std::move(params))}, defaults) {}

                torch::Tensor step(LossClosure closure = nullptr) override;
                void save(serialize::OutputArchive& archive) const override;
                void load(serialize::InputArchive& archive) override;

            private:
                template <typename Self, typename Archive>
                static void serialize(Self& self, Archive& archive) {
                    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(CMAES);
                }
        };
    }
}