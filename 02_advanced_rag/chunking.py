"""
Recursive Boundary-Aware Chunking — Advanced Approach.

Splits text hierarchically by paragraphs, newlines, and sentences
before falling back to character boundaries. Keeps table rows and
prose sentences coherent.
"""

from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class AdvancedChunk:
    """A boundary-aware chunk with rich metadata."""
    chunk_id: str           # Unique identifier: e.g. "nvidia_sec_10k_2024.txt#4"
    text: str               # The actual text content
    source_file: str        # Filename
    chunk_index: int        # Sequential index
    char_count: int         # Length in characters



def recursive_split_text(
    text: str,
    max_chunk_size: int=600,
    chunk_overlap: int=100, 
    delimiters: list[str] | None=None
)-> list[str]:
    """
    Recursively split text using the strongest available delimiter first.
    
    Order of preference:
    1. "\\n\\n" -> Paragraphs and markdown blocks
    2. "\\n"   -> Table rows and lists
    3. ". "    -> Full sentences
    4. " "     -> Word boundaries
    """
    if delimiters is None:
        delimiters=["\n\n", "\n", ". ", " "]

    # Base case: text is already within target size
    if len(text)<=max_chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the highest-priority delimiter present in the text
    chosen_delim=None
    for delim in delimiters:
        if delim in text:
            chosen_delim=delim
            break

    
    # If no delimiter found, hard-slice with overlap
    if chosen_delim is None:
        slices=[]
        step=max_chunk_size-chunk_overlap
        for i in range(0, len(text), step):
            slice_text=text[i: i+max_chunk_size].strip()
            if slice_text:
                slices.append(slice_text)

        return slices

    # Split by the chosen delimiter
    raw_pieces=text.split(chosen_delim)
    merged_chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for piece in raw_pieces:
        piece=piece.strip()
        if not piece:
            continue 

        piece_len=len(piece) + len(chosen_delim)

        # if adding this piece exceeds max_chunk_size, flush current chunk
        if current_lenght + piece_len> max_chunk_size and current_chunk:
            combined=chosen_delim.join(current_chunk).strip()
            if combined:
                merged_chunks.append(combined)

            # Implement overlap: keep the tail of the current chunk
            overlap_buffer=[]
            overlap_len=0
            for prev_piece in reversed(current_chunk):
                if overlap_len + len(prev_piece)<=chunk_overlap:
                    overlap_buffer.insert(0, prev_piece)
                    overlap_len+=len(prev_piece)
                else:
                    break

            
            current_chunk = overlap_buffer
            current_length = overlap_len

        # If a single piece is itself larger than max_chunk_size, recursively split it
        # with the next lower priority delimiter
        if piece_len> max_chunk_size:
            next_delimiters = delimiters[delimiters.index(chosen_delim) + 1:]
            sub_chunks=recursive_split_text(
                piece, max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                delimiters=next_delimiters,
            )
            merged_chunks.extend(sub_chunks)
        else:
            current_chunk.append(piece)
            current_length+=piece_len

    if current_chunk:
        combined=chosen_delim.join(current_chunk).strip()
        if combined:
            merged_chunks.append(combined)

    return merged_chunks



def load_and_chunk_corpus(data_dir: Path, max_chunk_size: int, chunk_overlap: int)-> list[AdvancedChunk]:
    """
    Load all files in data/ and chunk them recursively with boundary preservation.
    """
    all_chunks: list[AdvancedChunk]=[]
    txt_files=sorted(data_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No .txt documents found in {data_dir}")

    total_idx=0
    for file_path in txt_files:
        content=file_path.read_text(encoding="utf-8")
        raw_splits=recursive_split_text(
            text=content,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

        for i, split_text in enumerate(raw_splits):
            chunk=AdvancedChunk(
                chunk_id=f"{file_path.name}#{i}",
                text=split_text,
                source_file=file_path.name,
                chunk_index=total_idx,
                char_count=len(split_text)
            )
            all_chunks.append(chunk)
            total_idx+=1

    
    print(f"[AdvancedChunking] Loaded {len(txt_files)} files -> {len(all_chunks)} boundary-safe chunks.")
    return all_chunks
