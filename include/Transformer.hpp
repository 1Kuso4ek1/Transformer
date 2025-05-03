#pragma once
#include <torch/nn/init.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/embedding.h>
#include <torch/nn/modules/transformer.h>
#include <torch/nn/modules/transformercoder.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/functional/activation.h>

#include <generator>

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
                            .dropout(0.2)
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

    std::generator<int64_t> generate(
        std::vector<int64_t>&& tokens,
        size_t maxSize, size_t maxTokens,
        float temperature = 0.7, int k = 10)
    {
        eval();
        to(global::device);

        torch::NoGradGuard noGrad;

        for(int i = 0; i < maxTokens; i++)
        {
            auto tensor =
                torch::tensor(tokens, torch::kInt64)
                    .unsqueeze(0);

            auto res = forward(tensor);
            auto token = sampleTopK(res, temperature, k);

            if(token == 2)
                break;

            tokens.push_back(token); // Used like a context

            if(tokens.size() > maxSize)
                tokens.erase(tokens.begin());

            co_yield token;
        }
    }

    torch::Tensor forward(torch::Tensor src)
    {
        auto pos = torch::arange(0, src.size(1), torch::kLong);
        pos = posEncoder(pos).unsqueeze(0);

        auto data = (embedding(src) + pos).permute({ 1, 0, 2 });

        auto paddingMask = (src == 0);
        auto memoryMask = torch::full({ src.size(1), src.size(1) }, -std::numeric_limits<float>::infinity());
        memoryMask.index_put_({ torch::indexing::Slice(), 0 }, 0.0f);

        auto tgtMask = torch::nn::TransformerImpl::generate_square_subsequent_mask(src.size(1));

        auto res =
            decoder->forward(
                data, data, tgtMask, memoryMask,
                paddingMask, paddingMask
            );

        res = linear(res.permute({ 1, 0, 2 }).contiguous());

        return res;
    }

private:
    int64_t sampleTopK(const torch::Tensor& res, float temperature, int64_t k)
    {
        auto last = res[0][res.size(1) - 1];

        for(auto& i : { 0, 1, 3, 4 }) // Forbidden tokens
            last[i] = -1e9;

        auto probs = torch::softmax(last / temperature, -1);
        auto topK = torch::topk(probs, k);

        probs.zero_();
        probs.index_put_({ std::get<1>(topK) }, std::get<0>(topK));

        probs = probs / probs.sum();

        auto token = torch::multinomial(probs, 1).item<int64_t>();

        return token;
    }

private:
    torch::nn::Embedding embedding{ nullptr };
    torch::nn::Embedding posEncoder{ nullptr };
    torch::nn::TransformerDecoder decoder{ nullptr };
    torch::nn::Linear linear{ nullptr };
};
