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
            learnByRolesNextTokenCompleteSequenceShiftedAndAlsoThisFunctionNameIsAnAbloluteMonstrosityGodHelpMePleaseImLosingMyMind(data, tokenizer);
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
                user = true;
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

    void learnByRolesNextToken(
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
                user = true;
                continue;
            }

            auto modified = separatePunctuation(i);

            if(user)
                rawData.push_back(context + " [USER] " + modified + " [ASSISTANT]");
            else
            {
                auto view = std::views::split(modified, ' ')
                    | std::ranges::to<std::vector<std::string>>();

                for(auto token = view.begin(); token < view.end(); token++)
                {
                    if(token == view.begin())
                        rawTargets.push_back(*token);

                    rawData.push_back(rawData.back() + ' ' + *token);

                    if(token == view.end() - 1)
                        rawTargets.push_back("[END]");
                    else if(!(token + 1)->empty())
                        rawTargets.push_back(*(token + 1));
                }
            }

            context += (user ? " [USER] " : " [ASSISTANT] ") + modified + ' ';

            user = !user;
        }
    }

    void learnByRolesNextTokenCompleteSequenceShiftedAndAlsoThisFunctionNameIsAnAbloluteMonstrosityGodHelpMePleaseImLosingMyMind(
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
                user = true;
                continue;
            }

            auto modified = separatePunctuation(i);

            if(user)
            {
                rawData.push_back(context + " [USER] " + modified + " [ASSISTANT]");
            }
            else
            {
                rawTargets.push_back(context + " [ASSISTANT] " + modified + " [END]");
                rawData.back() += ' ' + modified + " [END]";
            }
            /* {
                auto view = std::views::split(rawData.back() + ' ' + modified, ' ')
                    | std::ranges::to<std::vector<std::string>>();

                std::string src, tgt;

                for(auto token = view.begin(); token < view.end(); token++)
                {
                    if(token > view.begin())
                        tgt += ' ' + *token;
                    if(token < view.end() - 1)
                        src += ' ' + *token;

                    if(token == view.end() - 1)
                        tgt += " [END]";
                }

                rawData.push_back(src);
                rawTargets.push_back(tgt);
            } */

            context += (user ? " [USER] " : " [ASSISTANT] ") + modified + ' ';

            user = !user;
        }
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
        auto seq = data[index];
        //auto tgt = index < targets.size() ? targets[index] : torch::zeros({ (long)maxSize }, torch::kInt64);

        /* std::cout << "Src: " << src << '\n';
        std::cout << "Tgt: " << tgt << '\n'; */
        
        auto src = seq.slice(0, 0, seq.size(0) - 1);
        auto tgt = seq.slice(0, 1, seq.size(0));

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
