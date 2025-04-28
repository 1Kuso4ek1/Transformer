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
            const auto modified = separatePunctuation(text);

            for(const auto& token : std::views::split(modified, ' '))
            {
                auto str = std::string(std::string_view(token));
                
                tokens.insert(str);
            }
        }

        int64_t id = 5;
        for(const auto& token : tokens)
            tokenIds[token] = id++;
    }

    std::vector<int64_t> encode(const std::string_view& text)
    {
        std::vector<int64_t> encoded;

        const auto modified = separatePunctuation(std::string(text));
        
        for(const auto& token : std::views::split(modified, ' '))
        {
            if(token.empty())
                continue;

            auto str = std::string(std::string_view(token));

            /* std::vector<int64_t> punctuation;

            if(str[0] != '[')
            {
                for(int i = 0; i < str.size(); i++)
                {
                    if(std::ispunct(str[i]))
                    {
                        punctuation.push_back(tokenIds[std::string(1, str[i])]);
                        str.erase(str.begin() + i);
                    }
                }
            } */

            auto it = tokenIds.find(str);

            if(it != tokenIds.end())
                encoded.push_back(it->second);/* 

            for(const auto& i : punctuation)
                encoded.push_back(i); */
        }

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
