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
                auto str = std::string(std::string_view(token));
                
                for(int i = 0; i < str.size(); i++)
                    if(std::ispunct(str[i]))
                    {
                        tokens.insert(std::string(1, str[i]));
                        str.erase(str.begin() + i);
                    }
                
                tokens.insert(str);
            }

        int64_t id = 2;
        for(const auto& token : tokens)
            tokenIds[token] = id++;
    }

    std::vector<int64_t> encode(const std::string_view& text)
    {
        std::vector<int64_t> encoded;
        
        for(const auto& token : std::views::split(text, ' '))
        {
            auto str = std::string(std::string_view(token));

            for(int i = 0; i < str.size(); i++)
                if(std::ispunct(str[i]))
                {
                    encoded.push_back(tokenIds[std::string(1, str[i])]);
                    str.erase(str.begin() + i);
                }

            auto it = tokenIds.find(str);

            if(it != tokenIds.end())
                encoded.push_back(it->second);
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
