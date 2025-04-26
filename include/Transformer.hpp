#pragma once
#include <torch/nn/init.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/transformer.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/functional/activation.h>

#include <Global.hpp>

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
        transformer =
            register_module(
                "transformer",
                torch::nn::Transformer(
                    torch::nn::TransformerOptions(dModel, 8, 4, 4)
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

        /* for(int i = 0; i < maxTokens; i++)
        {
            auto tensor =
                torch::tensor(tokens, torch::kInt64)
                    .unsqueeze(0);

            auto res = forward(tensor);

            // rewrite
            auto probs = torch::softmax(res[-1].squeeze(0) / temperature, -1); */
            /* torch::Tensor index;
            try
            {
                index = torch::multinomial(probs, 1);
            }
            catch(...)
            {
                continue;
            } */

            /* tokens.erase(tokens.begin());
            tokens.push_back(0); */

            /* auto firstZero = std::find(tokens.begin(), tokens.end(), 0);

            for(int i = 0; i < probs.size(-1) && firstZero + i < tokens.end(); i++)
                if((probs[i].item<int64_t>() != 0))
                {
                    *(firstZero++) = (probs[i].item<int64_t>());
                    output.push_back(probs[i].item<int64_t>());
                    
                    //if(output.back() == (index[i + 1].item<int64_t>()))
                        break;
                }
        } */

        for(const auto& i : output)
            std::cout << i << ' ';
        std::cout << '\n';

        return output;
    }

    // rewrite
    torch::Tensor forward(torch::Tensor src)
    {
        auto posEncoderMask = (src != 0).to(torch::kFloat32).unsqueeze(-1);

        auto pos = torch::arange(0, src.size(1), torch::kLong);
        pos = posEncoder(pos).unsqueeze(0) * posEncoderMask;

        auto data = (embedding(src) + pos).permute({ 1, 0, 2 });

        auto srcMask = (src == 0);

        auto res =
            transformer->forward(
                data, data,
                {}, {}, {},
                srcMask//srcMask
            );

        res =
            //torch::nn::functional::relu(
                linear(res.permute({ 1, 0, 2 }).contiguous());
            //);

        return res;
    }

private:
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Embedding posEncoder{ nullptr };
    torch::nn::Transformer transformer{ nullptr };
    torch::nn::Linear linear{ nullptr };
};
