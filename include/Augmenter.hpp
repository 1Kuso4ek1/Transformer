#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <ranges>

#include <Utils.hpp>

class Augmenter
{
public:
    Augmenter(
        const std::vector<std::string>& data,
        const std::vector<std::string>& augmentations
    )
    {
        for(const auto& i : augmentations)
            parseAugmentations(i);

        // 8 variations of dialogues
        for(int i = 0; i < 8; i++)
        {
            for(const auto& j : data)
            {
                augmented.emplace_back();

                const auto modified = separatePunctuation(j);

                const auto view = std::views::split(modified, ' ')
                    | std::ranges::to<std::vector<std::string>>();

                for(const auto& word : view)
                {
                    auto newWord = word;

                    if(synonyms.count(word) && !synonyms[word].empty())
                    {
                        if(rand() % 2 == 0)
                            newWord = synonyms[word][rand() % synonyms[word].size()];
                    }

                    augmented.back() += newWord;

                    if(newWord != "[RESET]")
                        augmented.back() += ' ';
                }
            }
        }
    }

    const std::vector<std::string>& getAugmented() const
    {
        return augmented;
    }
    
private:
    void parseAugmentations(const std::string& augmentation)
    {
        auto colon = augmentation.find(':');
        if(colon != std::string::npos)
        {
            std::string word = augmentation.substr(0, colon);
            std::stringstream stream(augmentation.substr(colon + 1));

            std::string synonym;
            while(std::getline(stream, synonym, ','))
                synonyms[word].push_back(synonym);
        }
    }

private:
    std::vector<std::string> augmented;
    std::unordered_map<std::string, std::vector<std::string>> synonyms;
};
