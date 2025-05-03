#pragma once
#include <memory>
#include <print>

#include <Utils.hpp>
#include <Tokenizer.hpp>

#include <ATen/core/TensorBody.h>
#include <torch/nn/functional/loss.h>

template<class Loader, class Network>
class Tester
{
public:
    Tester(Loader&& loader, std::shared_ptr<Network> network)
        : loader(std::move(loader)), network(network)
    {}

    void test(const Tokenizer& tokenizer)
    {
        std::println("Testing...");

        network->eval();
        network->to(global::device);

        torch::NoGradGuard noGrad;

        for(const auto& batch : *loader)
        {
            auto res = network->forward(batch.data);

            auto probs = res[-1].argmax(-1);

            std::cout << "Test: ";
            for(int i = 0; i < batch.data.size(-1); i++)
                if(batch.data[-1][i].template item<int64_t>() != 0)
                    std::cout << tokenizer.decode(batch.data[-1][i].template item<int64_t>()) << ' ';
            std::cout << "\n\n";

            std::cout << "Predicted: ";
            for(int i = 0; i < probs.size(-1); i++)
            {
                auto item = probs[i].template item<int64_t>();
                if(item != 0/*  && item != 2 */)
                    std::cout << tokenizer.decode(item) << ' ';
            }
            std::cout << "\n\n";
        }
    }

private:
    Loader loader;
    std::shared_ptr<Network> network;
};
