#pragma once
#include <vector>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <print>

class Tokenizer
{
public:
    void tokenize(const std::vector<std::string>& data)
    {
        for(const auto& text : data)
            for(const auto& token : std::views::split(text, ' '))
            {
                std::println("{}", std::string_view(token));
                tokens.insert(std::string(std::string_view(token)));
            }

        int64_t id = 2;
        for(const auto& token : tokens)
            tokenIds[token] = id++;
    }

    std::vector<int64_t> encode(const std::string_view& text)
    {
        std::vector<int64_t> encoded;
        
        for(const auto& token : std::views::split(text, ' '))
            encoded.push_back(tokenIds[std::string(std::string_view(token))]);

        return encoded;
    }

    std::string decode(int64_t token)
    {
        for(const auto& [key, val] : tokenIds)
            if(val == token)
                return key;
            
        return "[UNK]";
    }

    size_t size() const
    {
        return tokens.size();
    }

private:
    std::unordered_set<std::string> tokens;
    std::unordered_map<std::string, int64_t> tokenIds =
    {
        { "[PAD]", 0 },
        { "[UNK]", 1 }
    };
};
