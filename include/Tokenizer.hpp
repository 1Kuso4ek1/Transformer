#pragma once
#include <vector>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <print>

#include <Utils.hpp>

class Tokenizer
{
public:
    void tokenize(const std::vector<std::string>& data)
    {
        for(const auto& text : data)
        {
            auto modified = separatePunctuation(text);
            modified = toLower(std::move(modified));

            for(const auto& token : std::views::split(modified, ' '))
            {
                auto str = std::string(token.begin(), token.end());
                
                tokens.insert(str);
            }
        }

        int64_t id = 5;
        for(const auto& token : tokens)
            tokenIds[token] = id++;
    }

    std::vector<int64_t> encode(const std::string_view& text) const
    {
        std::vector<int64_t> encoded;

        auto modified = separatePunctuation(std::string(text));
        modified = toLower(std::move(modified));
        
        for(const auto& token : std::views::split(modified, ' '))
        {
            if(token.empty())
                continue;

            auto str = std::string(std::string_view(token));

            auto it = tokenIds.find(str);

            if(it != tokenIds.end())
                encoded.push_back(it->second);
        }

        return encoded;
    }

    std::string decode(int64_t token) const
    {
        for(const auto& [key, val] : tokenIds)
            if(val == token)
                return key;
            
        return "[UNK]";
    }

    size_t size() const
    {
        return tokenIds.size();
    }

private:
    std::unordered_set<std::string> tokens;
    std::unordered_map<std::string, int64_t> tokenIds =
    {
        { "[PAD]", 0 },
        { "[UNK]", 1 },
        { "[END]", 2 },
        { "[USER]", 3 },
        { "[ASSISTANT]", 4 }
    };
};
