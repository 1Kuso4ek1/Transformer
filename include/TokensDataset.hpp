#pragma once
#include <torch/data.h>

#include <Tokenizer.hpp>

class TokensDataset : public torch::data::Dataset<TokensDataset>
{
public:
    TokensDataset(
        const std::vector<std::string>& data,
        Tokenizer& tokenizer,
        size_t maxSize
    ) : maxSize(maxSize)
    {
        for(const auto& i : data)
        {
            std::string modified;

            for(int j = 0; j < i.size(); j++)
            {
                if(std::ispunct(i[j]) && !modified.empty() && modified.back() != ' ')
                    modified += ' ';
                
                modified += i[j];
            }

            auto view = std::views::split(modified, ' ')
                | std::ranges::to<std::vector<std::string>>();

            for(auto token = view.begin(); token < view.end() - 1; token++)
            {
                rawData.push_back(token == view.begin() ? *token : rawData.back() + ' ' + *token);
                rawTargets.push_back(*(token + 1));
            }
        }

        for(const auto& i : rawData)
        {
            auto tokens = tokenizer.encode(i);
            tokens.resize(maxSize, 0);
            this->data.push_back(torch::tensor(tokens, torch::kInt64));
        }

        for(const auto& i : rawTargets)
        {
            auto tokens = tokenizer.encode(i);
            tokens.resize(maxSize, 0);
            this->targets.push_back(torch::tensor(tokens, torch::kInt64));
        }       
    }

    torch::data::Example<> get(size_t index) override
    {
        auto src = data[index];
        auto tgt = index < targets.size() ? targets[index] : torch::zeros({ (long)maxSize }, torch::kInt64);

        /* std::cout << "Src: " << src << '\n';
        std::cout << "Tgt: " << tgt << '\n'; */
        
        /* torch::Tensor src = seq.slice(0, 0, seq.size(0) - 1);
        torch::Tensor tgt = seq.slice(0, 1, seq.size(0)); */

        return { src, tgt };
    }

    torch::optional<size_t> size() const override
    {
        return data.size();
    }

private:
    size_t maxSize;
    std::vector<std::string> rawData, rawTargets;
    std::vector<torch::Tensor> data, targets;
};
