#pragma once
#include <torch/data.h>

#include <Tokenizer.hpp>

class TokensDataset : public torch::data::Dataset<TokensDataset>
{
public:
    TokensDataset(
        const std::vector<std::string>& data,
        Tokenizer& tokenizer,
        size_t maxSize,
        bool roles
    ) : maxSize(maxSize)
    {
        if(roles)
            learnByRoles(data, tokenizer);
        else
            learnNextToken(data, tokenizer);

        encodeRawData(tokenizer);
    }

    void learnNextToken(
        const std::vector<std::string>& data,
        Tokenizer& tokenizer
    )
    {
        for(const auto& i : data)
        {
            auto modified = separatePunctuation(i);

            auto view = std::views::split(modified, ' ')
                | std::ranges::to<std::vector<std::string>>();

            for(auto token = view.begin(); token < view.end() - 1; token++)
            {
                rawData.push_back(token == view.begin() ? *token : rawData.back() + ' ' + *token);
                rawTargets.push_back(*(token + 1));
            }
        }
    }

    void learnByRoles(
        const std::vector<std::string>& data,
        Tokenizer& tokenizer
    )
    {
        bool user = true;

        std::string context;

        for(const auto& i : data)
        {
            if(i == "[RESET]")
            {
                context.clear();
                continue;
            }

            auto modified = separatePunctuation(i);
            
            if(user)
                rawData.push_back(context + " [USER] " + modified);
            else
                rawTargets.push_back(modified + " [END]");

            context += (user ? " [USER] " : " [ASSISTANT] ") + modified + ' ';

            user = !user;
        }
    }

    std::string separatePunctuation(const std::string& str)
    {
        std::string modified;

        for(int j = 0; j < str.size(); j++)
        {
            if(std::ispunct(str[j]) && !modified.empty() && modified.back() != ' ')
                modified += ' ';
            
            modified += str[j];
        }

        return modified;
    }

    void encodeRawData(Tokenizer& tokenizer)
    {
        for(const auto& i : rawData)
        {
            auto tokens = tokenizer.encode(i);
            
            if(tokens.size() > maxSize)
                tokens.erase(tokens.begin(), tokens.begin() + tokens.size() - maxSize);
            
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
