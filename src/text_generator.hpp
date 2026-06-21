#pragma once

#include "markov_chain.hpp"
#include "pos_tagger.hpp"
#include "word_association_classifier.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

class TextGenerator {
public:
    TextGenerator() : tagger("./udpipe", "models/english.udpipe"), rng(std::random_device{}()) {}

    void loadDataset(const std::string& dataDir, const std::string& embeddingsFile = "") {
        if (!embeddingsFile.empty()) {
            if (classifier.loadEmbeddings(embeddingsFile))
                std::cout << "  Loaded embeddings (" << classifier.dimension() << "d, "
                          << classifier.vocabSize() << " words)\n";
            else
                std::cout << "  Warning: could not load embeddings\n";
        }

        if (!fs::exists(dataDir)) return;

        for (const auto& entry : fs::directory_iterator(dataDir)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".txt") continue;

            std::string genre = entry.path().stem().string();
            std::ifstream file(entry.path());
            std::vector<std::string> lines;
            std::string line;

            while (std::getline(file, line)) {
                if (!line.empty()) {
                    chains[genre].train(line);
                    lines.push_back(line);
                }
            }
            classifier.loadGenre(genre, lines);
            lore[genre] = std::move(lines);
            std::cout << "  Loaded '" << genre << "' (" << lore[genre].size() << " phrases)\n";
        }
        std::cout << "Loaded " << chains.size() << " genres.\n";

        buildSwapCandidates();
    }

    void loadBlueprint(const std::string& path) {
        if (!fs::exists(path)) {
            std::cout << "  No blueprint directory, skipping flow model\n";
            return;
        }
        for (const auto& entry : fs::directory_iterator(path)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".txt") continue;
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) blueprintChain.train(line);
            }
        }
        std::cout << "  Loaded blueprint flow model\n";
    }

    std::string generate(const std::string& genre, int wordCount,
                         double swapProb = 0.15) const {
        auto it = chains.find(genre);
        if (it == chains.end() || it->second.empty()) return "";

        std::uniform_real_distribution<double> roll(0.0, 1.0);
        std::vector<std::string> sentences;
        std::unordered_set<std::string> usedSentences;
        int totalWords = 0;
        int targetWords = std::max(wordCount, 8);

        int maxAttempts = std::max(200, targetWords * 30);
        int attempts = 0;
        while (totalWords < targetWords && attempts < maxAttempts) {
            ++attempts;
            std::vector<std::string> sw;
            std::vector<std::string> orig;
            Prefix cur = it->second.randomStart();
            sw.push_back(cur.first);
            sw.push_back(cur.second);
            orig.push_back(cur.first);
            orig.push_back(cur.second);

            for (int step = 0; step < 24; ++step) {
                if (!it->second.hasPrefix(cur)) break;
                std::string next = it->second.randomNext(cur);
                if (next == MarkovChain::END_TOKEN) break;

                std::string out = next;
                if (roll(rng) < swapProb) {
                    auto sit = swaps.find(next);
                    if (sit != swaps.end() && !sit->second.empty()) {
                        std::uniform_int_distribution<size_t> p(0, sit->second.size() - 1);
                        std::string cand = sit->second[p(rng)];
                        if (it->second.wordProb(cur, cand) > 0 || roll(rng) < 0.15)
                            out = cand;
                    }
                }
                sw.push_back(out);
                orig.push_back(next);
                cur = {cur.second, next};
            }

            if (sw.size() < 10) continue;
            while (!sw.empty() && isDangling(sw.back())) { sw.pop_back(); orig.pop_back(); }
            if (sw.size() < 8) continue;

            bool hasRepeat = false;
            for (size_t i = 1; i < sw.size(); ++i)
                if (sw[i] == sw[i-1]) { hasRepeat = true; break; }
            if (hasRepeat) continue;

            {
                std::string origText = orig[0];
                for (size_t oi = 1; oi < orig.size(); ++oi)
                    origText += " " + orig[oi];
                if (sentencePerplexity(genre, origText) > 60.0) continue;
            }

            {
                std::string checkText = sw[0];
                for (size_t ci = 1; ci < sw.size(); ++ci)
                    checkText += " " + sw[ci];
                auto scores = classifier.getScores(checkText);
                bool genreMatch = false;
                for (const auto& [g, s] : scores)
                    if (g == genre && s > 0) { genreMatch = true; break; }
                if (!genreMatch) continue;
            }

            if (roll(rng) < 0.5) enhanceDescriptive(sw, genre);

            {
                std::string checkText = sw[0];
                for (size_t ci = 1; ci < sw.size(); ++ci)
                    checkText += " " + sw[ci];
                auto scores = classifier.getScores(checkText);
                bool genreMatch = false;
                for (const auto& [g, s] : scores)
                    if (g == genre && s > 0) { genreMatch = true; break; }
                if (!genreMatch) continue;
            }

            polishSentence(sw, genre);

            {
                std::string flowText = sw[0];
                for (size_t fi = 1; fi < sw.size(); ++fi)
                    flowText += " " + sw[fi];
                if (combinedPerplexity(genre, flowText) > 55.0) continue;
            }

            sw[0][0] = static_cast<char>(std::toupper(static_cast<unsigned char>(sw[0][0])));
            std::string sentence = sw[0];
            for (size_t i = 1; i < sw.size(); ++i)
                sentence += " " + sw[i];
            sentence += ".";

            if (usedSentences.count(sentence)) continue;
            usedSentences.insert(sentence);

            sentences.push_back(sentence);
            totalWords += static_cast<int>(sw.size());
        }

        if (sentences.empty()) return "";
        joinSentences(sentences, genre);

        std::string result = sentences[0];
        for (size_t i = 1; i < sentences.size(); ++i)
            result += " " + sentences[i];
        result = polishNouns(result, genre);
        // Only run WFC passes on shorter texts to avoid OOM
        size_t wc = 0;
        {
            std::istringstream cnt(result);
            std::string _;
            while (cnt >> _) ++wc;
        }
        if (wc >= 8 && wc <= 35) {
            result = wfcExpand(result, genre);
            result = wfcDelete(result, genre);
        }
        return result;
    }

    double sentencePerplexity(const std::string& genre, const std::string& text) const {
        auto it = chains.find(genre);
        if (it == chains.end()) return 100.0;
        auto words = tokenize(text);
        if (words.size() < 3) return 100.0;
        double totalLogProb = 0.0;
        int known = 0, unknown = 0;
        for (size_t i = 2; i < words.size(); ++i) {
            Prefix p{words[i-2], words[i-1]};
            double prob = it->second.wordProb(p, words[i]);
            if (prob > 0.0) {
                totalLogProb += std::log(prob);
                ++known;
            } else {
                ++unknown;
            }
        }
        if (known == 0) return 100.0;
        double avgLogProb = (totalLogProb - unknown * 5.0) / (known + unknown);
        return std::exp(-avgLogProb);
    }

    double flowPerplexity(const std::string& text) const {
        if (blueprintChain.empty()) return 50.0;
        auto words = tokenize(text);
        if (words.size() < 3) return 50.0;
        double totalLogProb = 0.0;
        int known = 0, unknown = 0;
        for (size_t i = 2; i < words.size(); ++i) {
            Prefix p{words[i-2], words[i-1]};
            double prob = blueprintChain.wordProb(p, words[i]);
            if (prob > 0.0) {
                totalLogProb += std::log(prob);
                ++known;
            } else {
                ++unknown;
            }
        }
        if (known == 0) return 50.0;
        double avgLogProb = (totalLogProb - unknown * 5.0) / (known + unknown);
        return std::exp(-avgLogProb);
    }

    double combinedPerplexity(const std::string& genre, const std::string& text) const {
        double cp = sentencePerplexity(genre, text);
        double fp = flowPerplexity(text);
        return CONTENT_WEIGHT * cp + FLOW_WEIGHT * fp;
    }

    std::vector<std::pair<std::string, double>> classify(const std::string& text) const {
        return classifier.getScores(text);
    }

    std::vector<std::string> getGenres() const {
        std::vector<std::string> result;
        for (const auto& [g, _] : chains) result.push_back(g);
        return result;
    }

    bool empty() const { return chains.empty(); }
    void setAddLore(bool v) { addLore = v; }
    POSTagger& getTagger() { return tagger; }

private:
    std::unordered_map<std::string, MarkovChain> chains;
    std::unordered_map<std::string, std::vector<std::string>> lore;
    WordAssociationClassifier classifier;
    std::unordered_map<std::string, std::vector<std::string>> swaps;
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::vector<double>>>> genreAdjectives;
    POSTagger tagger;
    mutable std::mt19937 rng;
    bool addLore = true;

    MarkovChain blueprintChain;
    static constexpr double CONTENT_WEIGHT = 0.35;
    static constexpr double FLOW_WEIGHT = 0.65;

    void joinSentences(std::vector<std::string>& sentences, const std::string& genre) const {
        if (sentences.size() < 2) return;

        struct Pattern {
            const char* str;
            bool prefix;
        };

        static const Pattern PATTERNS[] = {
            {" where ", false}, {" which ", false}, {" while ", false}, {" when ", false},
            {" as ", false}, {" until ", false}, {" though ", false},
            {", where ", false}, {", which ", false}, {", while ", false}, {", when ", false},
            {", as ", false}, {", until ", false}, {", though ", false},
            {", and ", false}, {", but ", false}, {", so ", false}, {", yet ", false},
            {"While ", true}, {"When ", true}, {"As ", true}, {"Although ", true},
            {"Because ", true}, {"Though ", true}, {"After ", true}, {"Before ", true},
            {"Until ", true},
        };

        for (size_t si = 1; si < sentences.size(); ++si) {
            std::string& prev = sentences[si - 1];
            const std::string& curr = sentences[si];
            if (prev.empty() || curr.empty()) continue;
            if (prev.back() != '.') continue;

            std::string base = prev.substr(0, prev.size() - 1);
            std::string lowerCurr = curr;
            if (!lowerCurr.empty())
                lowerCurr[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(lowerCurr[0])));

            struct Candidate {
                std::string text;
                double score;
            };
            std::vector<Candidate> cands;
            double bestScore = std::numeric_limits<double>::max();

            for (const auto& p : PATTERNS) {
                std::string joined;
                if (p.prefix) {
                    std::string lowerBase = base;
                    if (!lowerBase.empty())
                        lowerBase[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(lowerBase[0])));
                    joined = std::string(p.str) + lowerBase + ", " + curr;
                } else {
                    joined = base + std::string(p.str) + lowerCurr;
                }
                double score = combinedPerplexity(genre, joined);
                if (score < bestScore) bestScore = score;
                if (score < 60.0) cands.push_back({joined, score});
            }

            if (cands.empty()) continue;

            std::vector<std::string> ok;
            for (const auto& c : cands)
                if (c.score < bestScore * 1.12)
                    ok.push_back(c.text);

            std::uniform_int_distribution<size_t> pick(0, ok.size() - 1);
            sentences[si - 1] = ok[pick(rng)];
            sentences.erase(sentences.begin() + si);
            --si;
        }
    }

    static std::string wordLower(const std::string& w) {
        std::string s = w;
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        while (!s.empty() && std::ispunct(static_cast<unsigned char>(s.back())))
            s.pop_back();
        return s;
    }

    static bool isNounTag(const std::string& pos) {
        return pos.size() >= 2 && pos[0] == 'N';
    }

    bool swapCreatesDoubleNoun(const std::vector<std::string>& words, size_t i,
                               const std::string& candFmt) const {
        // Check if the swap would put two nouns adjacent
        if (i > 0) {
            std::string prev = wordLower(words[i - 1]);
            std::string cand = wordLower(candFmt);
            auto pp = posCache.find(prev);
            auto cp = posCache.find(cand);
            if (pp != posCache.end() && cp != posCache.end()
                && isNounTag(pp->second) && isNounTag(cp->second))
                return true;
        }
        if (i + 1 < words.size()) {
            std::string cand = wordLower(candFmt);
            std::string next = wordLower(words[i + 1]);
            auto cp = posCache.find(cand);
            auto np = posCache.find(next);
            if (cp != posCache.end() && np != posCache.end()
                && isNounTag(cp->second) && isNounTag(np->second))
                return true;
        }
        return false;
    }

    std::string polishNouns(const std::string& text, const std::string& genre) const {
        if (swaps.empty()) return text;
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string w;
        while (iss >> w) words.push_back(w);
        if (words.size() < 5) return text;

        for (int pass = 0; pass < 3; ++pass) {
            bool changed = false;
            for (size_t i = 0; i < words.size(); ++i) {
                std::string clean = wordLower(words[i]);
                if (clean.empty()) continue;

                auto pit = posCache.find(clean);
                if (pit == posCache.end() || !isNounTag(pit->second)) continue;

                auto sit = swaps.find(clean);
                if (sit == swaps.end() || sit->second.empty()) continue;

                double basePpl = combinedPerplexity(genre, text);
                std::string bestWord = words[i];

                for (const auto& cand : sit->second) {
                    if (cand == clean) continue;
                    std::string candFmt = cand;
                    if (std::isupper(static_cast<unsigned char>(words[i][0])))
                        candFmt[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(cand[0])));

                    if (swapCreatesDoubleNoun(words, i, candFmt)) continue;

                    std::string modified = words[0];
                    for (size_t j = 1; j < words.size(); ++j)
                        modified += " " + (j == i ? candFmt : words[j]);

                    double newPpl = combinedPerplexity(genre, modified);
                    if (newPpl < basePpl * 0.995) {
                        basePpl = newPpl;
                        bestWord = candFmt;
                    }
                }

                if (bestWord != words[i]) {
                    words[i] = bestWord;
                    changed = true;
                }
            }
            if (!changed) break;
        }

        std::string result = words[0];
        for (size_t i = 1; i < words.size(); ++i)
            result += " " + words[i];
        return result;
    }

    std::string wfcExpand(const std::string& text, const std::string& genre) const {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string w;
        while (iss >> w) words.push_back(w);
        if (words.size() < 4 || words.size() > 50) return text;

        auto windowText = [&](const std::vector<std::string>& wrds,
                              size_t center, int radius) -> std::string {
            int start = static_cast<int>(center) - radius;
            if (start < 0) start = 0;
            size_t end = std::min(center + static_cast<size_t>(radius), wrds.size() - 1);
            if (start > static_cast<int>(end)) return "";
            std::string r = wrds[start];
            for (int k = start + 1; k <= static_cast<int>(end); ++k)
                r += " " + wrds[k];
            return r;
        };

        static const std::unordered_set<std::string> SKIP_LEFT = {
            "the", "a", "an", "this", "that", "these", "those",
            "and", "but", "or", "yet", "so", "nor", "for",
            "although", "though", "because", "since", "while", "when",
            "whenever", "wherever", "whereas", "unless", "until", "after",
            "before", "once", "if", "whether"
        };
        auto skipAt = [&](size_t i) -> bool {
            std::string lc = wordLower(words[i]);
            if (lc.empty()) return true;
            if (lc.back() == '.' || lc.back() == ',') return true;
            if (SKIP_LEFT.count(lc)) return true;
            return false;
        };

        size_t totalInserted = 0;
        for (int pass = 0; pass < 3 && totalInserted < 12; ++pass) {
            bool changed = false;
            for (size_t i = 0; i + 1 < words.size() && words.size() < 80; ++i) {
                if (skipAt(i) || skipAt(i + 1)) continue;

                double basePpl = combinedPerplexity(genre,
                    windowText(words, i, 4));

                std::string best;
                double bestPpl = basePpl;
                std::string left = wordLower(words[i]);
                std::string right = wordLower(words[i + 1]);
                auto pRight = posCache.find(right);
                bool rightIsNoun = pRight != posCache.end() && isNounTag(pRight->second);

                if (rightIsNoun) {
                    static const char* DETS[] = {"the", "a", "this", "no"};
                    for (auto d : DETS) {
                        words.insert(words.begin() + i + 1, d);
                        double ppl = combinedPerplexity(genre, windowText(words, i + 1, 4));
                        if (ppl < bestPpl * 0.96) { bestPpl = ppl; best = d; }
                        words.erase(words.begin() + i + 1);
                    }
                }

                auto pLeft = posCache.find(left);
                if (pLeft != posCache.end() && pLeft->second[0] == 'V' && rightIsNoun) {
                    static const char* PREPS[] = {"in", "on", "to", "from", "with", "by", "for", "of"};
                    for (auto p : PREPS) {
                        words.insert(words.begin() + i + 1, p);
                        double ppl = combinedPerplexity(genre, windowText(words, i + 1, 4));
                        if (ppl < bestPpl * 0.96) { bestPpl = ppl; best = p; }
                        words.erase(words.begin() + i + 1);
                    }
                }

                bool leftIsNoun = pLeft != posCache.end() && isNounTag(pLeft->second);
                if ((leftIsNoun && rightIsNoun) ||
                    (pLeft != posCache.end() && pLeft->second[0] == 'V'
                     && pRight != posCache.end() && pRight->second[0] == 'V')) {
                    static const char* CONJS[] = {"and", "but", "or"};
                    for (auto c : CONJS) {
                        words.insert(words.begin() + i + 1, c);
                        double ppl = combinedPerplexity(genre, windowText(words, i + 1, 4));
                        if (ppl < bestPpl * 0.96) { bestPpl = ppl; best = c; }
                        words.erase(words.begin() + i + 1);
                    }
                }

                if (!best.empty()) {
                    words.insert(words.begin() + i + 1, best);
                    changed = true;
                    ++totalInserted;
                    ++i;
                }
            }
            if (!changed) break;
        }

        std::string result = words[0];
        for (size_t i = 1; i < words.size(); ++i)
            result += " " + words[i];
        return result;
    }

    std::string wfcDelete(const std::string& text, const std::string& genre) const {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string w;
        while (iss >> w) words.push_back(w);
        if (words.size() < 4 || words.size() > 60) return text;

        auto makeText = [&](const std::vector<std::string>& w) -> std::string {
            if (w.empty()) return "";
            std::string r = w[0];
            for (size_t k = 1; k < w.size(); ++k) r += " " + w[k];
            return r;
        };

        for (int pass = 0; pass < 3 && words.size() >= 6; ++pass) {
            double baseline = combinedPerplexity(genre, makeText(words));
            std::vector<std::pair<size_t, double>> scores;
            for (size_t i = 0; i < words.size(); ++i) {
                std::string clean = wordLower(words[i]);
                if (clean.empty() || clean.back() == '.') continue;
                if (i > 0 && wordLower(words[i - 1]).back() == '.') continue;
                std::string saved = words[i];
                words.erase(words.begin() + i);
                double ppl = combinedPerplexity(genre, makeText(words));
                words.insert(words.begin() + i, saved);
                if (ppl < baseline) scores.push_back({i, ppl});
            }

            if (scores.empty()) break;

            std::sort(scores.begin(), scores.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            size_t maxDel = words.size() / 8 + 1;
            size_t deleted = 0;
            for (const auto& s : scores) {
                if (deleted >= maxDel) break;
                size_t idx = s.first - deleted;
                std::string cln = wordLower(words[idx]);
                if (cln == "the" || cln == "a" || cln == "an" || cln == "this" || cln == "that") continue;
                words.erase(words.begin() + idx);
                ++deleted;
            }
            if (deleted == 0) break;
        }

        return makeText(words);
    }

    void polishSentence(std::vector<std::string>& sw, const std::string& genre) const {
        auto cit = chains.find(genre);
        if (cit == chains.end() || sw.size() < 3) return;
        const auto& chain = cit->second;

        auto computeCombinedPpl = [&](const std::vector<std::string>& w) -> double {
            double contentTotal = 0.0;
            int contentKnown = 0, contentUnknown = 0;
            for (size_t i = 2; i < w.size(); ++i) {
                double prob = chain.wordProb(Prefix{w[i-2], w[i-1]}, w[i]);
                if (prob > 0.0) {
                    contentTotal += std::log(prob);
                    ++contentKnown;
                } else {
                    ++contentUnknown;
                }
            }
            double cp = 100.0;
            if (contentKnown > 0) {
                double avgCP = (contentTotal - contentUnknown * 5.0) / (contentKnown + contentUnknown);
                cp = std::exp(-avgCP);
            }

            double fp = 50.0;
            if (!blueprintChain.empty() && contentKnown > 0) {
                double flowTotal = 0.0;
                int flowKnown = 0, flowUnknown = 0;
                for (size_t i = 2; i < w.size(); ++i) {
                    Prefix pw{w[i-2], w[i-1]};
                    double prob = blueprintChain.wordProb(pw, w[i]);
                    if (prob > 0.0) {
                        flowTotal += std::log(prob);
                        ++flowKnown;
                    } else {
                        ++flowUnknown;
                    }
                }
                if (flowKnown > 0) {
                    double avgLP = (flowTotal - flowUnknown * 5.0) / (flowKnown + flowUnknown);
                    fp = std::exp(-avgLP);
                }
            }

            return CONTENT_WEIGHT * cp + FLOW_WEIGHT * fp;
        };

        for (int pass = 0; pass < 3; ++pass) {
            double basePpl = computeCombinedPpl(sw);
            bool improved = false;

            for (size_t i = 2; i < sw.size(); ++i) {
                auto pit = posCache.find(sw[i]);
                if (pit == posCache.end() || !isContent(pit->second)) continue;

                auto sit = swaps.find(sw[i]);
                if (sit == swaps.end() || sit->second.empty()) continue;

                std::string bestWord = sw[i];
                double bestPpl = basePpl;

                for (const auto& cand : sit->second) {
                    if (cand == sw[i]) continue;
                    // Skip swap if it creates adjacent nouns
                    if (i > 0 && isNounTag(posCache[sw[i-1]]) && isNounTag(posCache[cand])) continue;
                    if (i + 1 < sw.size() && isNounTag(posCache[cand]) && isNounTag(posCache[sw[i+1]])) continue;

                    std::string orig = sw[i];
                    sw[i] = cand;
                    double newPpl = computeCombinedPpl(sw);
                    if (newPpl < bestPpl * 0.99) {
                        bestPpl = newPpl;
                        bestWord = cand;
                    }
                    sw[i] = orig;
                }

                if (bestWord != sw[i]) {
                    sw[i] = bestWord;
                    basePpl = bestPpl;
                    improved = true;
                }
            }

            if (!improved) break;
        }
    }

    void enhanceDescriptive(std::vector<std::string>& sw, const std::string& genre) const {
        auto adjs = genreAdjectives.find(genre);
        if (adjs == genreAdjectives.end() || adjs->second.empty()) return;

        std::vector<size_t> nounPos;
        for (size_t i = 0; i < sw.size(); ++i) {
            auto pit = posCache.find(sw[i]);
            if (pit != posCache.end() && pit->second.size() >= 2 && pit->second[0] == 'N')
                nounPos.push_back(i);
        }
        if (nounPos.empty()) return;

        std::uniform_int_distribution<size_t> pickPos(0, nounPos.size() - 1);
        size_t idx = nounPos[pickPos(rng)];
        const std::string& noun = sw[idx];

        auto eit = classifier.getEmbeddings().find(noun);
        if (eit == classifier.getEmbeddings().end()) return;

        std::vector<std::pair<double, std::string>> candidates;
        for (const auto& [adj, adjVec] : adjs->second) {
            double sim = 0;
            for (size_t d = 0; d < eit->second.size(); ++d)
                sim += eit->second[d] * adjVec[d];
            if (sim > 0.2) candidates.emplace_back(sim, adj);
        }

        if (candidates.empty()) return;
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        size_t n = std::min<size_t>(5, candidates.size());
        std::uniform_int_distribution<size_t> pick(0, n - 1);
        sw.insert(sw.begin() + idx, candidates[pick(rng)].second);
    }

    void buildSwapCandidates() {
        if (classifier.dimension() == 0) return;
        std::cout << "  Building word swap index...\n";

        std::vector<std::string> allLines;
        for (const auto& [g, lines] : lore)
            for (const auto& line : lines)
                allLines.push_back(line);

        auto tokens = tagger.tagLines(allLines);
        size_t ti = 0;
        for (const auto& line : allLines) {
            auto lineWords = tokenize(line);
            for (const auto& w : lineWords) {
                if (ti < tokens.size() && classifier.hasEmbedding(w)) {
                    auto it = posCache.find(w);
                    if (it == posCache.end())
                        posCache[w] = tokens[ti].pos;
                }
                ++ti;
            }
        }

        const auto& emb = classifier.getEmbeddings();
        std::vector<std::pair<std::string, std::vector<double>>> vocab;
        for (const auto& [w, _] : emb) {
            auto pit = posCache.find(w);
            if (pit != posCache.end() && isContent(pit->second))
                vocab.emplace_back(w, emb.at(w));
        }

        for (size_t i = 0; i < vocab.size(); ++i) {
            const auto& [word, vec] = vocab[i];
            const auto& pos = posCache[word];
            std::vector<std::pair<double, std::string>> cands;
            for (size_t j = 0; j < vocab.size(); ++j) {
                if (i == j) continue;
                if (pos[0] != posCache[vocab[j].first][0]) continue;
                double sim = 0;
                for (size_t d = 0; d < vec.size(); ++d)
                    sim += vec[d] * vocab[j].second[d];
                if (sim > 0.65) cands.emplace_back(sim, vocab[j].first);
            }
            std::sort(cands.begin(), cands.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            auto& s = swaps[word];
            for (size_t k = 0; k < std::min<size_t>(6, cands.size()); ++k)
                s.push_back(cands[k].second);
        }
        std::cout << "    " << swaps.size() << " words with swap candidates\n";

        std::cout << "  Precomputing genre adjectives...\n";
        for (const auto& [genre, lines] : lore) {
            auto& adjList = genreAdjectives[genre];
            std::unordered_set<std::string> seen;
            for (const auto& line : lines) {
                auto words = tokenize(line);
                for (const auto& w : words) {
                    auto pit = posCache.find(w);
                    if (pit == posCache.end() || pit->second.size() < 2 || pit->second[0] != 'A') continue;
                    if (seen.count(w)) continue;
                    seen.insert(w);
                    auto eit = emb.find(w);
                    if (eit != emb.end()) adjList.emplace_back(w, eit->second);
                }
            }
            std::cout << "    " << genre << ": " << adjList.size() << " adjectives\n";
        }
    }

    static bool isContent(const std::string& pos) {
        return pos.size() >= 2 && (pos[0] == 'N' || pos[0] == 'V' || pos[0] == 'A' || pos[0] == 'J');
    }

    static double cosineSim(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0;
        for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
        return dot;
    }

    static bool isDangling(const std::string& w) {
        return w == "the" || w == "a" || w == "an" || w == "of" || w == "in"
            || w == "to" || w == "for" || w == "with" || w == "at" || w == "on"
            || w == "by" || w == "from" || w == "into" || w == "through"
            || w == "along" || w == "about" || w == "as" || w == "or"
            || w == "and" || w == "but" || w == "if" || w == "then"
            || w == "so" || w == "up" || w == "down" || w == "out"
            || w == "over" || w == "under" || w == "again" || w == "further"
            || w == "then" || w == "once" || w == "here" || w == "there"
            || w == "when" || w == "where" || w == "why" || w == "how"
            || w == "all" || w == "each" || w == "every" || w == "both"
            || w == "few" || w == "some" || w == "any" || w == "only";
    }

    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string w;
        while (iss >> w) {
            std::transform(w.begin(), w.end(), w.begin(), ::tolower);
            while (!w.empty() && std::ispunct(static_cast<unsigned char>(w.front())))
                w.erase(w.begin());
            while (!w.empty() && std::ispunct(static_cast<unsigned char>(w.back())))
                w.pop_back();
            if (!w.empty())
                words.push_back(w);
        }
        return words;
    }

    mutable std::unordered_map<std::string, std::string> posCache;
};
