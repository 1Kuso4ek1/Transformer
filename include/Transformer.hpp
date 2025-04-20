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
    Transformer(size_t vocabSize, size_t dModel)
    {
        embedding =
            register_module(
                "embedding",
                torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(vocabSize + 2, dModel)
                        .padding_idx(0)
                )
            );
        transformer =
            register_module(
                "transformer",
                torch::nn::Transformer(
                    torch::nn::TransformerOptions(dModel, 2, 1, 1)
                )
            );
        linear = register_module("linear", torch::nn::Linear(dModel, vocabSize + 2));

        torch::nn::init::xavier_uniform_(embedding->weight);
        torch::nn::init::xavier_uniform_(linear->weight);
    }

    std::vector<int64_t> generate(std::vector<int64_t>&& tokens, size_t maxSize, size_t maxTokens)
    {
        std::vector<int64_t> output;
        output.reserve(maxTokens);

        tokens.resize(maxSize, 0);

        eval();
        to(global::device);

        torch::NoGradGuard noGrad;

        for(int i = 0; i < maxTokens; i++)
        {
            auto tensor =
                torch::tensor(tokens, torch::kInt64)
                    .unsqueeze(0);

            auto res = forward(tensor, tensor);

            auto index = (res[-1].squeeze(0)).argmax(-1);

            auto firstZero = std::find(tokens.begin(), tokens.end(), 0);

            for(int i = 0; i < index.size(-1) && firstZero + i < tokens.end(); i++)
                if((index[i].item<int64_t>() != 0))
                {
                    *(firstZero++) = (index[i].item<int64_t>());
                    output.push_back(index[i].item<int64_t>());
                    
                    if(output.back() == (index[i + 1].item<int64_t>()))
                        break;
                }
        }

        return output;
    }

    // I'll probably remove the tgt completelly. Or am I?
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt)
    {
        auto data = embedding(src).permute({ 1, 0, 2 });
        //auto target = embedding(tgt).permute({ 1, 0, 2 });

        auto srcMask = (src == 0);
        //auto tgtMask = (tgt == 0);
        /* auto mask =
            torch::nn::TransformerImpl::generate_square_subsequent_mask(
                data.size(0)
            ); */

        auto res =
            transformer->forward(
                data, data,
                {}, {}, {},
                srcMask, srcMask
            );

        res =
            torch::nn::functional::relu(
                linear(res.permute({ 1, 0, 2 }).contiguous())
            );

        return res;
    }

private:
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Transformer transformer{ nullptr };
    torch::nn::Linear linear{ nullptr };
};
