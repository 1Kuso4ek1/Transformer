#pragma once
#include <memory>
#include <print>

#include <Config.hpp>
#include <Utils.hpp>

#include <ATen/core/TensorBody.h>
#include <torch/nn/functional/loss.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/optim/adam.h>
#include <torch/optim/schedulers/step_lr.h>
#include <torch/serialize.h>

template<class Loader, class Network>
class Trainer
{
public:
    Trainer(
        Loader&& loader,
        std::shared_ptr<Network> network,
        const Config& config
    ) : loader(std::move(loader)),
        network(network),
        config(config),
        optimizer(
            network->parameters(),
            torch::optim::AdamOptions(config.learningRate)
            .betas({ 0.9, 0.98 })
            .eps(1e-9)
        )
    {
        if(config.load)
            torch::load(optimizer, config.optimizerPath);
    }

    void train()
    {
        network->train();
        network->to(global::device);

        std::println("Training...");

        for(int epoch = 0; epoch < config.epochs; epoch++)
        {
            at::Tensor loss{};

            for(const auto& batch : *loader)
            {
                optimizer.zero_grad();

                auto data = batch.data.to(global::device);
                auto target = batch.target.to(global::device);

                // Forward pass
                auto output = network->forward(data);
                
                loss =
                    torch::nn::functional::cross_entropy(
                        output.view({ -1, output.size(-1) }),
                        target.view(-1),
                        torch::nn::CrossEntropyLossOptions()
                            .ignore_index(0)
                    );

                // Backward pass
                if(loss.requires_grad())
                {
                    loss.backward();
                    torch::nn::utils::clip_grad_norm_(network->parameters(), 0.5);
                    optimizer.step();
                }
            }

            if((epoch + 1) % 10 == 0)
            {
                static auto modelFilename = std::filesystem::path(config.modelPath).stem().string();
                static auto optFilename = std::filesystem::path(config.optimizerPath).stem().string();

                torch::save(network, std::format("{}_epoch_{}.pt", modelFilename, epoch + 1));
                torch::save(optimizer, std::format("{}_epoch_{}.pt", optFilename, epoch + 1));
            }

            std::println("Epoch: {}\tLoss: {}", epoch + 1, loss.item<float>());
        }

        torch::save(network, config.modelPath);
        torch::save(optimizer, config.optimizerPath);
    }

private:
    Loader loader;
    std::shared_ptr<Network> network;

    const Config& config;

    torch::optim::Adam optimizer;
};
