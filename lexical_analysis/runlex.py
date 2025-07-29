from tqdm import tqdm
import os 
import sys
import regex
lexicons = {}

def import_lexicon(file):
    global lexicons
    with open(file, 'r') as f:
        lexicon = f.readlines()
    lexicon = set(word.strip().lower() for word in lexicon if word.strip())
    lexicons[file] = lexicon

def analyze(transcript):
    global lexicons
    lines = transcript.split('\n')
    lines_with_hit = 0
    results = []
    for line in lines:
        has_hit = False
        for file, lexicon in lexicons.items():
            for word in lexicon:
                if regex.search(r'\b' + regex.escape(word) + r'\b', line, regex.IGNORECASE):
                    results.append((file, word, line.strip()))
                    lines_with_hit += 1 if not has_hit else 0
                    has_hit = True
                    
    return results, lines_with_hit

def import_lexicons(directory):
    global lexicons
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        sys.exit(1)
    
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            import_lexicon(os.path.join(directory, file))


def run_lexical_analysis(transcripts):
    global lexicons
    results = []
    for transcript in tqdm(transcripts, desc="Analyzing transcripts"):
        linehits, result = analyze(transcript)
        results.append((linehits, result))
    return zip(*results) if results else ([], [])

def load_transcripts(directory):
    transcripts = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        sys.exit(1)
    
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r') as f:
                transcripts.append(f.read())
    return transcripts





if __name__ == "__main__":
    word_frequencies = {}
    lexicon_hits = {}


    if len(sys.argv) < 3:
        print("Usage: python runlex.py <lexicon_directory> <transcript_directory>")
        sys.exit(1)
    
    lexicon_directory = sys.argv[1]
    transcript_directory = sys.argv[2]
    
    import_lexicons(lexicon_directory)
    transcripts = load_transcripts(transcript_directory)
    num_lines = [len(transcript.split('\n')) for transcript in transcripts]
    line_hits, results = run_lexical_analysis(transcripts)
    line_hit_ratios = [sum(hits) / float(lines) for hits, lines in zip(line_hits, num_lines)]

    transcript_ratios = sum([1 for hit in line_hits if len(hit) > 0]) / float(len(transcripts)) # average number of transcripts with hits from the lexicons


    for result in results:
        print(result)

    
    for file, word, line in result:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1
        lexicon_hits[file] = lexicon_hits.get(file, 0) + 1

    
    # print all the data using tabulate
    from tabulate import tabulate

    print("\nLexicon Hits:")
    lexicon_hits_table = [[file, count] for file, count in lexicon_hits.items()]
    print(tabulate(lexicon_hits_table, headers=["Lexicon File", "Hits"], tablefmt="grid"))

    print("\nWord Frequencies:")
    word_frequencies_table = [[word, count] for word, count in word_frequencies.items()]
    print(tabulate(word_frequencies_table, headers=["Word", "Frequency"], tablefmt="grid"))

    print(f"\nTotal Lines Analyzed: {sum(num_lines)}")
    print(f"Total # of Lines with Hits: {sum(line_hits)}")
    print(f"Avg # of lines with Hits: {(sum(line_hits) / float(sum(num_lines))):.3f}")
    print(f"Total Transcripts: {len(transcripts)}")
    print(f"Transcripts with Hits: {sum(1 for hit in line_hits if len(hit) > 0)}")
    print(f"% Transcripts with Hits: {transcript_ratios:.3f}")
    print(f"Total Unique Words: {len(word_frequencies)}")

    # Print word frequencies in descending order
    print("\nWord Frequencies (Descending):")
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_word_frequencies:
        print(f"{word}: {count}")
    print("\nLexicon Hits in Descending Order:")
    sorted_lexicon_hits = sorted(lexicon_hits.items(), key=lambda x: x[1], reverse=True)
    for file, count in sorted_lexicon_hits:
        print(f"{file}: {count}")


