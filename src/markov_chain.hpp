#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using Prefix = std::pair<std::string, std::string>;

struct PrefixHash {
    size_t operator()(const Prefix& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class MarkovChain {
public:
    static constexpr const char* END_TOKEN = "\x02END\x02";

    MarkovChain() : rng(std::random_device{}()) {}

    explicit MarkovChain(unsigned seed) : rng(seed) {}

    void train(const std::string& line) {
        auto words = tokenize(line);
        if (words.size() < 2) return;

        startPrefixes.emplace_back(words[0], words[1]);

        for (size_t i = 0; i + 2 < words.size(); ++i) {
            Prefix p{words[i], words[i + 1]};
            chain[p].push_back(words[i + 2]);
        }

        Prefix endP{words[words.size() - 2], words[words.size() - 1]};
        chain[endP].push_back(END_TOKEN);
    }

    std::string generate(int wordCount) const {
        if (chain.empty() || startPrefixes.empty()) return "";

        std::vector<std::string> words;
        Prefix current = startPrefixes[rng() % startPrefixes.size()];
        words.push_back(current.first);
        words.push_back(current.second);

        while (words.size() < static_cast<size_t>(wordCount)) {
            auto it = chain.find(current);
            if (it == chain.end() || it->second.empty()) {
                current = startPrefixes[rng() % startPrefixes.size()];
                words.push_back(current.first);
                words.push_back(current.second);
                continue;
            }

            const auto& choices = it->second;
            const std::string& next = choices[rng() % choices.size()];

            if (next == END_TOKEN) {
                current = startPrefixes[rng() % startPrefixes.size()];
                words.push_back(current.first);
                words.push_back(current.second);
                continue;
            }

            words.push_back(next);
            current = {current.second, next};
        }

        if (words.size() > static_cast<size_t>(wordCount))
            words.resize(wordCount);

        if (words.empty()) return "";

        words[0][0] = static_cast<char>(std::toupper(static_cast<unsigned char>(words[0][0])));
        std::string result = words[0];
        for (size_t i = 1; i < words.size(); ++i)
            result += " " + words[i];
        result += ".";

        return result;
    }

    std::string generateFrom(const std::string& seed1, const std::string& seed2, int wordCount) const {
        if (chain.empty()) return "";

        Prefix current{seed1, seed2};
        auto it = chain.find(current);
        if (it == chain.end() || it->second.empty())
            return generate(wordCount);

        std::vector<std::string> words;
        words.push_back(current.first);
        words.push_back(current.second);

        while (words.size() < static_cast<size_t>(wordCount)) {
            it = chain.find(current);
            if (it == chain.end() || it->second.empty()) break;

            const auto& choices = it->second;
            const std::string& next = choices[rng() % choices.size()];

            if (next == END_TOKEN) break;

            words.push_back(next);
            current = {current.second, next};
        }

        if (words.empty()) return "";

        words[0][0] = static_cast<char>(std::toupper(static_cast<unsigned char>(words[0][0])));
        std::string result = words[0];
        for (size_t i = 1; i < words.size(); ++i)
            result += " " + words[i];
        result += ".";

        return result;
    }

    Prefix randomStart() const {
        if (startPrefixes.empty()) return {};
        return startPrefixes[rng() % startPrefixes.size()];
    }

    bool hasPrefix(const Prefix& p) const {
        auto it = chain.find(p);
        return it != chain.end() && !it->second.empty();
    }

    size_t continuationCount(const Prefix& p) const {
        auto it = chain.find(p);
        if (it == chain.end()) return 0;
        return it->second.size();
    }

    std::string randomNext(const Prefix& p) const {
        auto it = chain.find(p);
        if (it == chain.end() || it->second.empty()) return END_TOKEN;
        return it->second[rng() % it->second.size()];
    }

    std::string nextAt(const Prefix& p, size_t idx) const {
        auto it = chain.find(p);
        if (it == chain.end() || idx >= it->second.size()) return END_TOKEN;
        return it->second[idx];
    }

    double wordProb(const Prefix& p, const std::string& word) const {
        auto it = chain.find(p);
        if (it == chain.end() || it->second.empty()) return 0.0;
        size_t count = 0;
        for (const auto& w : it->second)
            if (w == word) ++count;
        return static_cast<double>(count) / it->second.size();
    }

    double logProb(const Prefix& p, const std::string& word) const {
        double prob = wordProb(p, word);
        return prob > 0 ? std::log(prob) : -100.0;
    }

    const std::vector<Prefix>& getStartPrefixes() const { return startPrefixes; }
    bool empty() const { return chain.empty(); }

private:
    std::unordered_map<Prefix, std::vector<std::string>, PrefixHash> chain;
    std::vector<Prefix> startPrefixes;
    mutable std::mt19937 rng;

    static std::vector<std::string> tokenize(const std::string& line) {
        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.front())))
                word.erase(word.begin());
            while (!word.empty() && std::ispunct(static_cast<unsigned char>(word.back())))
                word.pop_back();
            if (!word.empty())
                words.push_back(word);
        }
        return words;
    }
};
