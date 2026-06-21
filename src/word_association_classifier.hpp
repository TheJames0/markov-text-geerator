#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class WordAssociationClassifier {
public:
    bool loadEmbeddings(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;

        std::string line;
        line.reserve(8192);

        while (std::getline(file, line)) {
            const char* start = line.data();
            const char* end = start + line.size();

            const char* sep = start;
            while (sep < end && *sep != ' ') ++sep;
            if (sep >= end) continue;

            std::string word(start, sep - start);
            if (word.empty()) continue;
            word = toLower(word);

            const char* ptr = sep + 1;
            std::vector<double> vec;
            vec.reserve(300);

            while (ptr < end) {
                while (ptr < end && *ptr == ' ') ++ptr;
                if (ptr >= end) break;
                double val;
                auto result = std::from_chars(ptr, end, val);
                if (result.ec != std::errc{}) break;
                vec.push_back(val);
                ptr = result.ptr;
            }

            if (!vec.empty()) {
                dim = vec.size();
                normalize(vec);
                embeddings[word] = std::move(vec);
            }
        }
        return !embeddings.empty();
    }

    void loadGenre(const std::string& genre, const std::vector<std::string>& lines) {
        genres.push_back(genre);

        for (const auto& line : lines) {
            auto words = tokenize(line);
            for (const auto& w : words)
                wordFreq[genre][w]++;
        }

        if (dim == 0) return;

        for (const auto& line : lines) {
            auto words = tokenize(line);
            for (const auto& w : words) {
                auto it = embeddings.find(w);
                if (it != embeddings.end())
                    genreWords[genre].push_back(it->second);
            }
        }

        if (genreWords[genre].empty()) return;

        std::vector<double> centroid(dim, 0.0);
        for (const auto& v : genreWords[genre]) {
            for (size_t i = 0; i < dim; ++i)
                centroid[i] += v[i];
        }
        for (size_t i = 0; i < dim; ++i)
            centroid[i] /= genreWords[genre].size();
        normalize(centroid);
        genreCentroids[genre] = std::move(centroid);
    }

    std::string classify(const std::string& text) const {
        auto scores = getScores(text);
        if (scores.empty()) return "";
        return scores.front().first;
    }

    std::vector<std::pair<std::string, double>> getScores(const std::string& text) const {
        auto words = tokenize(text);
        if (words.empty() || genres.empty()) return {};

        if (dim > 0 && !genreCentroids.empty())
            return getVotingScores(words);

        return getFreqScores(words);
    }

    std::vector<std::string> getGenres() const {
        return genres;
    }

    bool empty() const { return genres.empty(); }
    bool embeddingsLoaded() const { return !embeddings.empty(); }
    size_t vocabSize() const { return embeddings.size(); }
    size_t dimension() const { return dim; }
    const std::unordered_map<std::string, std::vector<double>>& getEmbeddings() const { return embeddings; }
    const std::vector<double>& getGenreCentroid(const std::string& genre) const {
        static const std::vector<double> empty;
        auto it = genreCentroids.find(genre);
        return it != genreCentroids.end() ? it->second : empty;
    }
    bool hasEmbedding(const std::string& word) const { return embeddings.count(word) > 0; }

    std::string wordGenre(const std::string& word) const {
        auto eit = embeddings.find(word);
        if (eit == embeddings.end() || dim == 0 || genres.empty()) return "";
        double bestSim = -1.0;
        std::string bestGenre;
        for (const auto& g : genres) {
            auto cit = genreCentroids.find(g);
            if (cit == genreCentroids.end()) continue;
            double sim = 0.0;
            for (size_t i = 0; i < dim; ++i)
                sim += eit->second[i] * cit->second[i];
            if (sim > bestSim) { bestSim = sim; bestGenre = g; }
        }
        return bestGenre;
    }

    double coherenceScore(const std::string& text, const std::string& targetGenre) const {
        auto words = tokenize(text);
        if (words.empty() || dim == 0) return 0.0;
        int relevant = 0, matching = 0;
        for (const auto& w : words) {
            auto eit = embeddings.find(w);
            if (eit == embeddings.end()) continue;
            std::string bestG = wordGenre(w);
            if (bestG.empty()) continue;
            ++relevant;
            if (bestG == targetGenre) ++matching;
        }
        return relevant > 0 ? (double)matching / relevant : 0.0;
    }

private:
    std::vector<std::string> genres;
    std::unordered_map<std::string, std::vector<double>> embeddings;
    std::unordered_map<std::string, std::vector<std::vector<double>>> genreWords;
    std::unordered_map<std::string, std::vector<double>> genreCentroids;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> wordFreq;
    size_t dim = 0;

    static void normalize(std::vector<double>& vec) {
        double norm = 0.0;
        for (size_t i = 0; i < vec.size(); ++i)
            norm += vec[i] * vec[i];
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t i = 0; i < vec.size(); ++i)
                vec[i] /= norm;
        }
    }

    double cosineSim(const std::vector<double>& a, const std::vector<double>& b) const {
        double dot = 0.0;
        for (size_t i = 0; i < dim; ++i)
            dot += a[i] * b[i];
        return dot;
    }

    std::vector<std::pair<std::string, double>> getVotingScores(
        const std::vector<std::string>& words
    ) const {
        size_t ng = genres.size();
        std::unordered_map<std::string, double> weightedScores;
        for (const auto& g : genres)
            weightedScores[g] = 0.0;

        double totalWeight = 0.0;
        for (const auto& w : words) {
            auto eit = embeddings.find(w);
            if (eit == embeddings.end()) continue;

            std::vector<double> sims(ng, 0.0);

            for (size_t gi = 0; gi < ng; ++gi) {
                const auto& genre = genres[gi];
                auto cit = genreCentroids.find(genre);
                if (cit == genreCentroids.end()) continue;
                double sim = cosineSim(eit->second, cit->second);

                auto gw = genreWords.find(genre);
                if (gw == genreWords.end()) continue;
                double bestWordSim = 0.0;
                for (const auto& wv : gw->second) {
                    double ws = cosineSim(eit->second, wv);
                    if (ws > bestWordSim) bestWordSim = ws;
                }

                sims[gi] = 0.5 * sim + 0.5 * bestWordSim;
            }

            size_t bestIdx = 0;
            for (size_t gi = 1; gi < ng; ++gi)
                if (sims[gi] > sims[bestIdx]) bestIdx = gi;

            double secondBest = 0.0;
            for (size_t gi = 0; gi < ng; ++gi) {
                if (gi != bestIdx && sims[gi] > secondBest)
                    secondBest = sims[gi];
            }

            double margin = sims[bestIdx] - secondBest;
            if (margin > 0.0) {
                weightedScores[genres[bestIdx]] += margin;
                totalWeight += margin;
            }
        }

        std::vector<std::pair<std::string, double>> scores;
        for (const auto& g : genres) {
            double score = totalWeight > 0
                ? weightedScores.at(g) / totalWeight
                : 0.0;
            scores.emplace_back(g, score);
        }

        std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        return scores;
    }

    std::vector<std::pair<std::string, double>> getFreqScores(
        const std::vector<std::string>& words
    ) const {
        std::vector<std::pair<std::string, double>> scores;

        std::unordered_map<std::string, int> textFreq;
        for (const auto& w : words)
            textFreq[w]++;

        for (const auto& genre : genres) {
            double score = 0;
            const auto& freq = wordFreq.at(genre);
            for (const auto& [word, count] : textFreq) {
                auto it = freq.find(word);
                if (it != freq.end())
                    score += std::log(1.0 + it->second) * count;
            }
            score /= static_cast<double>(words.size());
            scores.emplace_back(genre, score);
        }

        std::sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        return scores;
    }

    static std::string toLower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }

    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
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
