#pragma once
#include <torch/nn/init.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/transformer.h>
#include <torch/nn/modules/transformercoder.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/functional/activation.h>

#include <Utils.hpp>

class Transformer : public torch::nn::Module
{
public:
    Transformer(size_t vocabSize, size_t dModel, size_t maxSeq)
    {
        embedding =
            register_module(
                "embedding",
                torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(vocabSize, dModel)
                        .padding_idx(0)
                )
            );
        posEncoder =
            register_module(
                "posEncoder",
                torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(maxSeq, dModel)
                )
            );
        decoder =
            register_module(
                "decoder",
                torch::nn::TransformerDecoder(
                    torch::nn::TransformerDecoderOptions(
                        torch::nn::TransformerDecoderLayerOptions(dModel, 8)
                            .dropout(0.1)
                            .activation(torch::nn::GELU()),
                        4
                    )
                )
            );
        linear = register_module("linear", torch::nn::Linear(dModel, vocabSize));

        torch::nn::init::xavier_uniform_(embedding->weight);
        torch::nn::init::xavier_uniform_(posEncoder->weight);
        torch::nn::init::xavier_uniform_(linear->weight);
    }

    std::vector<int64_t> generate(const std::vector<int64_t>& tokens, size_t maxSize, size_t maxTokens, float temperature = 0.7)
    {
        std::vector<int64_t> output;
        output.reserve(maxTokens);

        eval();
        to(global::device);

        torch::NoGradGuard noGrad;

        auto tensor =
            torch::tensor(tokens, torch::kInt64)
                .unsqueeze(0);

        auto res = forward(tensor);

        auto probs = torch::softmax(res[-1].squeeze(0) / temperature, -1);
        probs = torch::multinomial(probs, 1);

        int index{};
        do
        {
            output.push_back(probs[index++].item<int64_t>());
        } while(output.back() != 2 && output.size() < maxTokens && index < probs.size(0));

        for(const auto& i : output)
            std::cout << i << ' ';
        std::cout << '\n';

        return output;
    }

    std::vector<int64_t> generateSequential(std::vector<int64_t>&& tokens, size_t maxSize, size_t maxTokens, float temperature = 0.7)
    {
        std::vector<int64_t> output;
        output.reserve(maxTokens);

        eval();
        to(global::device);

        torch::NoGradGuard noGrad;

        auto lastContextZero = std::find(tokens.begin(), tokens.end(), 0);

        for(int i = 0; i < maxTokens && lastContextZero < tokens.end(); i++)
        {
            auto tensor =
                torch::tensor(tokens, torch::kInt64)
                    .unsqueeze(0);

            auto res = forward(tensor);

            auto probs = torch::softmax(res[-1].squeeze(0) / temperature, -1);
            probs = torch::multinomial(probs, 1);

            auto token = probs[0].item<int64_t>();

            if(token == 2)
                break;

            *(lastContextZero++) = token;
            output.push_back(token);

            if(tokens.size() > maxSize)
                tokens.erase(tokens.begin());
        }

        for(const auto& i : output)
            std::cout << i << ' ';
        std::cout << '\n';

        return output;
    }

    torch::Tensor forward(torch::Tensor src)
    {
        auto pos = torch::arange(0, src.size(1), torch::kLong);
        pos = posEncoder(pos).unsqueeze(0);

        auto data = (embedding(src) + pos).permute({ 1, 0, 2 });

        auto paddingMask = (src == 0);
        auto tgtMask = torch::nn::TransformerImpl::generate_square_subsequent_mask(src.size(1));

        auto res =
            decoder->forward(
                data, data, {}, {},
                paddingMask
            );

        res = linear(res.permute({ 1, 0, 2 }).contiguous());

        return res;
    }

private:
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Embedding posEncoder{ nullptr };
    torch::nn::TransformerDecoder decoder{ nullptr };
    torch::nn::Linear linear{ nullptr };
};
