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
package org.apache.lucene.internal.vectorization;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorSpecies;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.store.RandomAccessInput;

/** Panama Vector API implementation of {@link DocValuesBulkDecodeSupport}. */
final class PanamaDocValuesBulkDecodeSupport implements DocValuesBulkDecodeSupport {

  static final PanamaDocValuesBulkDecodeSupport INSTANCE = new PanamaDocValuesBulkDecodeSupport();

  private static final VectorSpecies<Byte> BYTE_SPECIES = ByteVector.SPECIES_PREFERRED;

  private PanamaDocValuesBulkDecodeSupport() {}

  @Override
  public void decodeByteAligned(
      byte[] bytes, int bytesOffset, int bitsPerValue, long[] values, int valuesOffset, int count) {
    if (bitsPerValue != Long.SIZE
        || ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN
        || BYTE_SPECIES.vectorByteSize() < 32) {
      DefaultDocValuesBulkDecodeSupport.INSTANCE.decodeByteAligned(
          bytes, bytesOffset, bitsPerValue, values, valuesOffset, count);
      return;
    }

    final int valuesPerVector = BYTE_SPECIES.vectorByteSize() / Long.BYTES;
    final int loopBound = count - count % valuesPerVector;
    int i = 0;
    for (; i < loopBound; i += valuesPerVector) {
      ByteVector.fromArray(BYTE_SPECIES, bytes, bytesOffset + i * Long.BYTES)
          .reinterpretAsLongs()
          .intoArray(values, valuesOffset + i);
    }
    if (i < count) {
      DefaultDocValuesBulkDecodeSupport.INSTANCE.decodeByteAligned(
          bytes, bytesOffset + i * Long.BYTES, bitsPerValue, values, valuesOffset + i, count - i);
    }
  }

  private static final ValueLayout.OfShort SHORT_LE =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfInt INT_LE =
      ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE =
      ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @Override
  public boolean decodeByteAlignedToSegment(
      RandomAccessInput slice,
      long srcByteOffset,
      int bitsPerValue,
      MemorySegment dst,
      long dstByteOffset,
      int count)
      throws IOException {
    if (slice instanceof MemorySegmentAccessInput == false) {
      // e.g. NIOFSDirectory-backed input; let the caller take the heap path.
      return false;
    }
    final MemorySegmentAccessInput msai = (MemorySegmentAccessInput) slice;
    if (count == 0) {
      return true;
    }
    final int bytesPerValue = bitsPerValue / Byte.SIZE;
    final long byteLen = (long) count * bytesPerValue;
    final MemorySegment src = msai.segmentSliceOrNull(srcByteOffset, byteLen);
    if (src == null) {
      // The requested range straddles mmap chunk boundaries.
      return false;
    }
    switch (bitsPerValue) {
      case Long.SIZE ->
          // Bit-identical little-endian 8-byte longs: straight memory copy.
          MemorySegment.copy(src, 0, dst, dstByteOffset, byteLen);
      case Byte.SIZE -> {
        for (int i = 0; i < count; i++) {
          dst.set(
              LONG_LE,
              dstByteOffset + i * 8L,
              Byte.toUnsignedLong(src.get(ValueLayout.JAVA_BYTE, i)));
        }
      }
      case Short.SIZE -> {
        for (int i = 0; i < count; i++) {
          dst.set(LONG_LE, dstByteOffset + i * 8L, Short.toUnsignedLong(src.get(SHORT_LE, i * 2L)));
        }
      }
      case Integer.SIZE -> {
        for (int i = 0; i < count; i++) {
          dst.set(LONG_LE, dstByteOffset + i * 8L, Integer.toUnsignedLong(src.get(INT_LE, i * 4L)));
        }
      }
      case 24 -> {
        // Wide (4-byte) reads never run past the slice end except for the last element,
        // which is assembled byte-by-byte.
        for (int i = 0; i < count - 1; i++) {
          dst.set(LONG_LE, dstByteOffset + i * 8L, src.get(INT_LE, i * 3L) & 0xFFFFFFL);
        }
        dst.set(LONG_LE, dstByteOffset + (count - 1) * 8L, tailValue(src, (count - 1) * 3L, 3));
      }
      case 40, 48, 56 -> {
        final long mask = -1L >>> (Long.SIZE - bitsPerValue);
        for (int i = 0; i < count - 1; i++) {
          dst.set(
              LONG_LE, dstByteOffset + i * 8L, src.get(LONG_LE, (long) i * bytesPerValue) & mask);
        }
        dst.set(
            LONG_LE,
            dstByteOffset + (count - 1) * 8L,
            tailValue(src, (long) (count - 1) * bytesPerValue, bytesPerValue));
      }
      default -> {
        return false;
      }
    }
    return true;
  }

  /** Assembles a little-endian value from the last {@code numBytes} bytes of {@code src}. */
  private static long tailValue(MemorySegment src, long offset, int numBytes) {
    long value = 0;
    for (int b = 0; b < numBytes; b++) {
      value |= Byte.toUnsignedLong(src.get(ValueLayout.JAVA_BYTE, offset + b)) << (b * Byte.SIZE);
    }
    return value;
  }
}
