/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.index;

import java.io.IOException;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.CompiledAutomaton;

/**
 * A per-document byte[] with presorted values. This is fundamentally an iterator over the int ord
 * values per document, with random access APIs to resolve an int ord to BytesRef.
 *
 * <p>Per-Document values in a SortedDocValues are deduplicated, dereferenced, and sorted into a
 * dictionary of unique values. A pointer to the dictionary value (ordinal) can be retrieved for
 * each document. Ordinals are dense and in increasing sorted order.
 */
public abstract class SortedDocValues extends DocValuesIterator {

  /** Sole constructor. (For invocation by subclass constructors, typically implicit.) */
  protected SortedDocValues() {}

  /**
   * Returns the ordinal for the current docID. It is illegal to call this method after {@link
   * #advanceExact(int)} returned {@code false}.
   *
   * @return ordinal for the document: this is dense, starts at 0, then increments by 1 for the next
   *     value in sorted order.
   */
  public abstract int ordValue() throws IOException;

  /**
   * Bulk retrieval of ordinals. This API helps reduce the performance impact of virtual function
   * calls.
   *
   * <p>This API behaves as if implemented as below, which is the default implementation:
   *
   * <pre><code class="language-java">
   * public void ordValues(int size, int[] docs, int[] ords, int defaultOrd) throws IOException {
   *   for (int i = 0; i &lt; size; ++i) {
   *     int doc = docs[i];
   *     int ord;
   *     if (advanceExact(doc)) {
   *       ord = ordValue();
   *     } else {
   *       ord = defaultOrd;
   *     }
   *     ords[i] = ord;
   *   }
   * }
   * </code></pre>
   *
   * <p><b>NOTE</b>: The {@code docs} array is required to be sorted in ascending order with no
   * duplicates.
   *
   * <p><b>NOTE</b>: This API doesn't allow callers to know which doc IDs have a value or not. If
   * you need to exclude documents that don't have a value for this field, then you could apply a
   * {@link org.apache.lucene.search.FieldExistsQuery} as a {@link
   * org.apache.lucene.search.BooleanClause.Occur#FILTER} clause. Another option is to fall back to
   * using {@link #advanceExact} and {@link #ordValue()} on ranges of doc IDs that may not be dense,
   * e.g.
   *
   * <pre><code class="language-java">
   * if (size > 0 &amp;&amp; values.advanceExact(docs[0]) &amp;&amp; values.docIDRunEnd() &gt; docs[size - 1]) {
   *   // use values#ordValues to retrieve ordinals
   * } else {
   *   // some docs may not have a value, use #advanceExact and #ordValue
   * }
   * </code></pre>
   *
   * @param size the number of ordinals to retrieve
   * @param docs the buffer of doc IDs whose ordinals should be looked up
   * @param ords the buffer of ordinals to fill
   * @param defaultOrd the ordinal to put in the buffer when a document doesn't have a value
   */
  public void ordValues(int size, int[] docs, int[] ords, int defaultOrd) throws IOException {
    ordValues(size, docs, 0, ords, 0, defaultOrd);
  }

  /**
   * Offset-aware variant of {@link #ordValues(int, int[], int[], int)}. Reads {@code size} doc IDs
   * starting at {@code docs[docsOffset]} and writes the corresponding ordinals starting at {@code
   * ords[ordsOffset]}. This follows the same convention as {@link System#arraycopy}.
   *
   * @param size the number of ordinals to retrieve
   * @param docs the buffer of doc IDs whose ordinals should be looked up
   * @param docsOffset first position in {@code docs} to read
   * @param ords the buffer of ordinals to fill
   * @param ordsOffset first position in {@code ords} to write
   * @param defaultOrd the ordinal to put in the buffer when a document doesn't have a value
   */
  public void ordValues(
      int size, int[] docs, int docsOffset, int[] ords, int ordsOffset, int defaultOrd)
      throws IOException {
    for (int di = docsOffset, oi = ordsOffset, end = docsOffset + size; di < end; di++, oi++) {
      int ord;
      if (advanceExact(docs[di])) {
        ord = ordValue();
      } else {
        ord = defaultOrd;
      }
      ords[oi] = ord;
    }
  }

  /**
   * Retrieves the value for the specified ordinal. The returned {@link BytesRef} may be re-used
   * across calls to {@link #lookupOrd(int)} so make sure to {@link BytesRef#deepCopyOf(BytesRef)
   * copy it} if you want to keep it around.
   *
   * @param ord ordinal to lookup (must be &gt;= 0 and &lt; {@link #getValueCount()})
   * @see #ordValue()
   */
  public abstract BytesRef lookupOrd(int ord) throws IOException;

  /**
   * Returns the number of unique values.
   *
   * @return number of unique values in this SortedDocValues. This is also equivalent to one plus
   *     the maximum ordinal.
   */
  public abstract int getValueCount();

  /**
   * If {@code key} exists, returns its ordinal, else returns {@code -insertionPoint-1}, like {@code
   * Arrays.binarySearch}.
   *
   * @param key Key to look up
   */
  public int lookupTerm(BytesRef key) throws IOException {
    int low = 0;
    int high = getValueCount() - 1;

    while (low <= high) {
      int mid = (low + high) >>> 1;
      final BytesRef term = lookupOrd(mid);
      int cmp = term.compareTo(key);

      if (cmp < 0) {
        low = mid + 1;
      } else if (cmp > 0) {
        high = mid - 1;
      } else {
        return mid; // key found
      }
    }

    return -(low + 1); // key not found.
  }

  /**
   * Returns a {@link TermsEnum} over the values. The enum supports {@link TermsEnum#ord()} and
   * {@link TermsEnum#seekExact(long)}.
   */
  public TermsEnum termsEnum() throws IOException {
    return new SortedDocValuesTermsEnum(this);
  }

  /**
   * Returns a {@link TermsEnum} over the values, filtered by a {@link CompiledAutomaton} The enum
   * supports {@link TermsEnum#ord()}.
   */
  public TermsEnum intersect(CompiledAutomaton automaton) throws IOException {
    TermsEnum in = termsEnum();
    switch (automaton.type) {
      case NONE:
        return TermsEnum.EMPTY;
      case ALL:
        return in;
      case SINGLE:
        return new SingleTermsEnum(in, automaton.term);
      case NORMAL:
        return new AutomatonTermsEnum(in, automaton);
      default:
        // unreachable
        throw new RuntimeException("unhandled case");
    }
  }
}
