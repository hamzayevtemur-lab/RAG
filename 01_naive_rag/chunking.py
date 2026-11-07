"""
Fixed-Size Character Chunking — The Naive Approach.

Slides a fixed window across raw text with overlap.
No awareness of sentence boundaries, tables, or sections.
This is intentionally simple to establish a baseline.
"""

from dataclasses import dataclass

@dataclass
class Chunk:
    """A single chunk of text with metadata for traceability."""
    text: str               # The actual chunk content
    source_file: str        # Which file this chunk came from
    chunk_index: int        # Position in the chunk sequence
    start_char: int         # Character offset in the original document
    end_char: int           # End character offset


def fixed_size_chunk(text: str, source_file: str, chunk_size: int, chunk_overlap: int)->list[Chunk]:
    """
    Split text into fixed-size character chunks with overlap.
    Args:
        text:          Raw document text.
        source_file:   Filename for traceability.
        chunk_size:    Number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    Returns:
        List of Chunk objects.
    """

    chunks=[]

    step=chunk_size-chunk_overlap
    start=0
    index=0

    while start< len(text):
        end=start+chunk_size
        chunk_text=text[start:end]

        # Skip chunks that are only whitespace
        if chunk_text.strip():
            chunks.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                chunk_index=index,
                start_char=start,
                end_char=min(end, len(text)),
            ))
            index+=1
        
        start+=step
    
    return chunks




def load_and_chunk_all(data_dir, chunk_size: int, chunk_overlap: int)->list[Chunk]:
    """
    Load every .txt file from data_dir and chunk them all.
    Args:
        data_dir:      Path to the data/ folder.
        chunk_size:    Characters per chunk.
        chunk_overlap: Overlap between chunks.
    Returns:
        Flat list of Chunk objects across all files.
    """

    all_chunks=[]

    # Sort for deterministic ordering across runs
    txt_files = sorted(data_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for file_path in txt_files:
        raw_text=file_path.read_text(encoding="utf-8")
        file_chunks=fixed_size_chunk(
            text=raw_text,
            source_file=file_path.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(file_chunks)

    print(f"[Chunking] Loaded {len(txt_files)} files → {len(all_chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")

    return all_chunks

