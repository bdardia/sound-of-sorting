/******************************************************************************
 * src/SortAlgo.cpp
 *
 * Implementations is many sorting algorithms.
 *
 * Note that these implementations may not be as good/fast as possible. Some
 * are modified so that the visualization is more instructive.
 *
 * Futhermore, some algorithms are annotated using the mark() and watch()
 * functions from SortArray. These functions add colors to the illustratation
 * and thereby makes the algorithm's visualization easier to explain.
 *
 ******************************************************************************
 * The algorithms in this file are copyrighted by the original authors. All
 * code is freely available.
 *
 * The source code added by myself (Timo Bingmann) and all modifications are
 * copyright (C) 2013-2014 Timo Bingmann <tb@panthema.net>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "SortAlgo.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <inttypes.h>

typedef ArrayItem value_type;

// inversion count limit for iterator instrumented algorithms
const unsigned int inversion_count_instrumented = 512;

const struct AlgoEntry g_algolist[] =
{
    { _("Selection Sort"), &SelectionSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Insertion Sort"), &InsertionSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Binary Insertion Sort"), &BinaryInsertionSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Merge Sort"), &MergeSort, UINT_MAX, 512,
      _("Merge sort which merges two sorted sequences into a shadow array,"
        "and then copies it back to the shown array.") },
    { _("Merge Sort (iterative)"), &MergeSortIterative, UINT_MAX, 512,
      _("Merge sort variant which iteratively merges "
        "subarrays of sizes of powers of two.") },
    { _("Quick Sort (LR ptrs)"), &QuickSortLR, UINT_MAX, UINT_MAX,
      _("Quick sort variant with left and right pointers.") },
    { _("Quick Sort (LL ptrs)"), &QuickSortLL, UINT_MAX, UINT_MAX,
      _("Quick sort variant from 3rd edition of CLRS: two pointers on left.") },
    { _("Quick Sort (ternary, LR ptrs)"), &QuickSortTernaryLR, UINT_MAX, UINT_MAX,
      _("Ternary-split quick sort variant, adapted from multikey quicksort by "
        "Bentley & Sedgewick: partitions \"=<?>=\" using two pairs of pointers "
        "at left and right, then copied to middle.") },
    { _("Quick Sort (ternary, LL ptrs)"), &QuickSortTernaryLL, UINT_MAX, UINT_MAX,
      _("Ternary-split quick sort variant: partitions \"<>?=\" using two "
        "pointers at left and one at right. Afterwards copies the \"=\" to middle.") },
    { _("Quick Sort (dual pivot)"), &QuickSortDualPivot, UINT_MAX, UINT_MAX,
      _("Dual pivot quick sort variant: partitions \"<1<2?>\" using three pointers, "
        "two at left and one at right.") },
    { _("Three-Way Quicksort"), &threeWayQuicksortMain, UINT_MAX, 512,
  	  _("Usually the fastest version of quicksort.") },
    { _("Bubble Sort"), &BubbleSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Cocktail Shaker Sort"), &CocktailShakerSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
	{ _("Cocktail-Merge (Quarters)"), &dualCocktailMerge, UINT_MAX, 512,
	_("Cocktail sorts each quarter, then merges the quarters.") },
    { _("Gnome Sort"), &GnomeSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Comb Sort"), &CombSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Shell Sort"), &ShellSort, UINT_MAX, 1024,
      wxEmptyString },
    { _("Heap Sort"), &HeapSort, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Smooth Sort"), &SmoothSort, UINT_MAX, 1024,
      wxEmptyString },
    { _("Odd-Even Sort"), &OddEvenSort, UINT_MAX, 1024,
      wxEmptyString },
    // older sequential implementation, which really makes little sense to do
    //{ _("Bitonic Sort"), &BitonicSort, UINT_MAX, UINT_MAX, wxEmptyString },
    { _("Batcher's Bitonic Sort"), &BitonicSortNetwork, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Batcher's Odd-Even Merge Sort"), &BatcherSortNetwork, UINT_MAX, UINT_MAX,
      wxEmptyString },
    { _("Cycle Sort"), &CycleSort, 512, UINT_MAX,
      wxEmptyString },
    { _("Radix Sort (LSD)"), &RadixSortLSD, UINT_MAX, 512,
      _("Least significant digit radix sort, which copies item into a shadow "
        "array during counting.") },
	{ _("Ben Radix"), &BenRadix, UINT_MAX, 512,
	_("My Radix.") },
	{ _("Radix LSD In-Place"), &inPlaceRadix, UINT_MAX, 512,
	_("Similar to American flag sort.") },
	{ _("Pancake Sort"), &pancakeSort, UINT_MAX, 512,
	_("Aims for the minimun number of inversions.") },
    { _("Radix Sort (MSD)"), &RadixSortMSD, UINT_MAX, UINT_MAX,
      _("Most significant digit radix sort, which permutes items in-place by walking cycles.") },
	{ _("Pigeonhole Sort"), &pigeonholeSort, UINT_MAX, 512,
	_("Works well when the number of elements and number of possible key values are similar.") },
    { _("std::sort (gcc)"), &StlSort, UINT_MAX, inversion_count_instrumented,
      wxEmptyString },
    { _("std::stable_sort (gcc)"), &StlStableSort, UINT_MAX, inversion_count_instrumented,
      wxEmptyString },
    { _("std::sort_heap (gcc)"), &StlHeapSort, UINT_MAX, inversion_count_instrumented,
      wxEmptyString },
    { _("Tim Sort"), &TimSort, UINT_MAX, inversion_count_instrumented,
      wxEmptyString },
	{ _("Flashsort"), &flashSortMain, UINT_MAX, 512,
	_("Will go quadratic if the distribution is skewed.") },
    { _("Block Merge Sort (WikiSort)"), &WikiSort, UINT_MAX, inversion_count_instrumented,
      _("An O(1) place O(n log n) time stable merge sort.") },
    { _("Bogo Sort"), &BogoSort, 10, UINT_MAX,
      wxEmptyString },
    { _("Bozo Sort"), &BozoSort, 10, UINT_MAX,
      wxEmptyString },
    { _("Stooge Sort"), &StoogeSort, 256, inversion_count_instrumented,
      wxEmptyString },
    { _("Slow Sort"), &SlowSort, 128, inversion_count_instrumented,
      wxEmptyString },
	{ _("Smart Sort"), &smartSort, UINT_MAX, 512,
	_("A combo sort I'm working on") },
	{ _("Gravity Sort"), &gravitySort, UINT_MAX, 512,
	_("A natural sorting algorithm of O(S) complexity, where S is the sum of input numbers.") },
	{ _("Adaptive Left Radix"), &adaptiveRadixLeft, UINT_MAX, 512,
	_("Similar to flashsort, extremely fast.") }
};

const size_t g_algolist_size = sizeof(g_algolist) / sizeof(g_algolist[0]);

const struct AlgoEntry* g_algolist_end = g_algolist + g_algolist_size;

// ****************************************************************************
// *** Selection Sort

void SelectionSort(SortArray& A)
{
    volatile ssize_t jMin = 0;
    A.watch(&jMin, 3);

    for (size_t i = 0; i < A.size()-1; ++i)
    {
        jMin = i;

        for (size_t j = i+1; j < A.size(); ++j)
        {
            if (A[j] < A[jMin]) {
                A.mark_swap(j, jMin);
                jMin = j;
            }
        }

        A.swap(i, jMin);

        // mark the last good element
        if (i > 0) A.unmark(i-1);
        A.mark(i);
    }
    A.unwatch_all();
}

// ****************************************************************************
// *** Insertion Sort

// swaps every time (keeps all values visible)
void InsertionSort(SortArray& A)
{
    for (size_t i = 1; i < A.size(); ++i)
    {
        value_type key = A[i];
        A.mark(i);

        ssize_t j = i - 1;
        while (j >= 0 && A[j] > key)
        {
            A.swap(j, j+1);
            j--;
        }

        A.unmark(i);
    }
}

// with extra item on stack
void InsertionSort2(SortArray& A)
{
    for (size_t i = 1; i < A.size(); ++i)
    {
        value_type tmp, key = A[i];
        A.mark(i);

        ssize_t j = i - 1;
        while (j >= 0 && (tmp = A[j]) > key)
        {
            A.set(j + 1, tmp);
            j--;
        }
        A.set(j + 1, key);

        A.unmark(i);
    }
}

// swaps every time (keeps all values visible)
void BinaryInsertionSort(SortArray& A)
{
    for (size_t i = 1; i < A.size(); ++i)
    {
        value_type key = A[i];
        A.mark(i);

        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (key <= A[mid])
                hi = mid;
            else
                lo = mid + 1;
        }

        // item has to go into position lo

        ssize_t j = i - 1;
        while (j >= lo)
        {
            A.swap(j, j+1);
            j--;
        }

        A.unmark(i);
    }
}

// ****************************************************************************
// *** Merge Sort (out-of-place with sentinels)

// by myself (Timo Bingmann)

void Merge(SortArray& A, size_t lo, size_t mid, size_t hi)
{
    // mark merge boundaries
    A.mark(lo);
    A.mark(mid,3);
    A.mark(hi-1);

    // allocate output
    std::vector<value_type> out(hi-lo);

    // merge
    size_t i = lo, j = mid, o = 0; // first and second halves
    while (i < mid && j < hi)
    {
        // copy out for fewer time steps
        value_type ai = A[i], aj = A[j];

        out[o++] = (ai < aj ? (++i, ai) : (++j, aj));
    }

    // copy rest
    while (i < mid) out[o++] = A[i++];
    while (j < hi) out[o++] = A[j++];

    ASSERT(o == hi-lo);

    A.unmark(mid);

    // copy back
    for (i = 0; i < hi-lo; ++i)
        A.set(lo + i, out[i]);

    A.unmark(lo);
    A.unmark(hi-1);
}

void MergeSort(SortArray& A, size_t lo, size_t hi)
{
    if (lo + 1 < hi)
    {
        size_t mid = (lo + hi) / 2;

        MergeSort(A, lo, mid);
        MergeSort(A, mid, hi);

        Merge(A, lo, mid, hi);
    }
}

void MergeSort(SortArray& A)
{
    return MergeSort(A, 0, A.size());
}

void MergeSortIterative(SortArray& A)
{
    for (size_t s = 1; s < A.size(); s *= 2)
    {
        for (size_t i = 0; i + s < A.size(); i += 2 * s)
        {
            Merge(A, i, i + s,
                  std::min(i + 2 * s, A.size()));
        }
    }
}

// ****************************************************************************
// *** Quick Sort Pivot Selection

QuickSortPivotType g_quicksort_pivot = PIVOT_FIRST;

// some quicksort variants use hi inclusive and some exclusive, we require it
// to be _exclusive_. hi == array.end()!
ssize_t QuickSortSelectPivot(SortArray& A, ssize_t lo, ssize_t hi)
{
    if (g_quicksort_pivot == PIVOT_FIRST)
        return lo;

    if (g_quicksort_pivot == PIVOT_LAST)
        return hi-1;

    if (g_quicksort_pivot == PIVOT_MID)
        return (lo + hi) / 2;

    if (g_quicksort_pivot == PIVOT_RANDOM)
        return lo + (rand() % (hi - lo));

    if (g_quicksort_pivot == PIVOT_MEDIAN3)
    {
        ssize_t mid = (lo + hi) / 2;

        // cases if two are equal
        if (A[lo] == A[mid]) return lo;
        if (A[lo] == A[hi-1] || A[mid] == A[hi-1]) return hi-1;

        // cases if three are different
        return A[lo] < A[mid]
            ? (A[mid] < A[hi-1] ? mid : (A[lo] < A[hi-1] ? hi-1 : lo))
            : (A[mid] > A[hi-1] ? mid : (A[lo] < A[hi-1] ? lo : hi-1));
    }

    return lo;
}

wxArrayString QuickSortPivotText()
{
    wxArrayString sl;

    sl.Add( _("First Item") );
    sl.Add( _("Last Item") );
    sl.Add( _("Middle Item") );
    sl.Add( _("Random Item") );
    sl.Add( _("Median of Three") );

    return sl;
}

// ****************************************************************************
// *** Quick Sort LR (in-place, pointers at left and right, pivot is middle element)

// by myself (Timo Bingmann), based on Hoare's original code

void QuickSortLR(SortArray& A, ssize_t lo, ssize_t hi)
{
    // pick pivot and watch
    volatile ssize_t p = QuickSortSelectPivot(A, lo, hi+1);

    value_type pivot = A[p];
    A.watch(&p, 2);

    volatile ssize_t i = lo, j = hi;
    A.watch(&i, 3);
    A.watch(&j, 3);

    while (i <= j)
    {
        while (A[i] < pivot)
            i++;

        while (A[j] > pivot)
            j--;

        if (i <= j)
        {
            A.swap(i,j);

            // follow pivot if it is swapped
            if (p == i) p = j;
            else if (p == j) p = i;

            i++, j--;
        }
    }

    A.unwatch_all();

    if (lo < j)
        QuickSortLR(A, lo, j);

    if (i < hi)
        QuickSortLR(A, i, hi);
}

void QuickSortLR(SortArray& A)
{
    return QuickSortLR(A, 0, A.size()-1);
}

// ****************************************************************************
// *** Quick Sort LL (in-place, two pointers at left, pivot is first element and moved to right)

// by myself (Timo Bingmann), based on CLRS' 3rd edition

size_t PartitionLL(SortArray& A, size_t lo, size_t hi)
{
    // pick pivot and move to back
    size_t p = QuickSortSelectPivot(A, lo, hi);

    value_type pivot = A[p];
    A.swap(p, hi-1);
    A.mark(hi-1);

    volatile ssize_t i = lo;
    A.watch(&i, 3);

    for (size_t j = lo; j < hi-1; ++j)
    {
        if (A[j] <= pivot) {
            A.swap(i, j);
            ++i;
        }
    }

    A.swap(i, hi-1);
    A.unmark(hi-1);
    A.unwatch_all();

    return i;
}

void QuickSortLL(SortArray& A, size_t lo, size_t hi)
{
    if (lo + 1 < hi)
    {
        size_t mid = PartitionLL(A, lo, hi);

        QuickSortLL(A, lo, mid);
        QuickSortLL(A, mid+1, hi);
    }
}

void QuickSortLL(SortArray& A)
{
    return QuickSortLL(A, 0, A.size());
}

// ****************************************************************************
// *** Quick Sort Ternary (in-place, two pointers at left, pivot is first element and moved to right)

// by myself (Timo Bingmann), loosely based on multikey quicksort by B&S

void QuickSortTernaryLR(SortArray& A, ssize_t lo, ssize_t hi)
{
    if (hi <= lo) return;

    int cmp;

    // pick pivot and swap to back
    ssize_t piv = QuickSortSelectPivot(A, lo, hi+1);
    A.swap(piv, hi);
    A.mark(hi);

    const value_type& pivot = A[hi];

    // schema: |p ===  |i <<< | ??? |j >>> |q === |piv
    volatile ssize_t i = lo, j = hi-1;
    volatile ssize_t p = lo, q = hi-1;

    A.watch(&i, 3);
    A.watch(&j, 3);

    for (;;)
    {
        // partition on left
        while (i <= j && (cmp = A[i].cmp(pivot)) <= 0)
        {
            if (cmp == 0) {
                A.mark(p,4);
                A.swap(i, p++);
            }
            ++i;
        }

        // partition on right
        while (i <= j && (cmp = A[j].cmp(pivot)) >= 0)
        {
            if (cmp == 0) {
                A.mark(q,4);
                A.swap(j, q--);
            }
            --j;
        }

        if (i > j) break;

        // swap item between < > regions
        A.swap(i++, j--);
    }

    // swap pivot to right place
    A.swap(i,hi);
    A.mark_swap(i,hi);

    ssize_t num_less = i - p;
    ssize_t num_greater = q - j;

    // swap equal ranges into center, but avoid swapping equal elements
    j = i-1; i = i+1;

    ssize_t pe = lo + std::min(p-lo, num_less);
    for (ssize_t k = lo; k < pe; k++, j--) {
        A.swap(k,j);
        A.mark_swap(k,j);
    }

    ssize_t qe = hi-1 - std::min(hi-1-q, num_greater-1); // one already greater at end
    for (ssize_t k = hi-1; k > qe; k--, i++) {
        A.swap(i,k);
        A.mark_swap(i,k);
    }

    A.unwatch_all();
    A.unmark_all();

    QuickSortTernaryLR(A, lo, lo + num_less - 1);
    QuickSortTernaryLR(A, hi - num_greater + 1, hi);
}

void QuickSortTernaryLR(SortArray& A)
{
    return QuickSortTernaryLR(A, 0, A.size()-1);
}

// ****************************************************************************
// *** Quick Sort LL (in-place, two pointers at left, pivot is first element and moved to right)

// by myself (Timo Bingmann)

std::pair<ssize_t,ssize_t> PartitionTernaryLL(SortArray& A, ssize_t lo, ssize_t hi)
{
    // pick pivot and swap to back
    ssize_t p = QuickSortSelectPivot(A, lo, hi);

    value_type pivot = A[p];
    A.swap(p, hi-1);
    A.mark(hi-1);

    volatile ssize_t i = lo, k = hi-1;
    A.watch(&i, 3);

    for (ssize_t j = lo; j < k; ++j)
    {
        int cmp = A[j].cmp(pivot); // ternary comparison
        if (cmp == 0) {
            A.swap(--k, j);
            --j; // reclassify A[j]
            A.mark(k,4);
        }
        else if (cmp < 0) {
            A.swap(i++, j);
        }
    }

    // unwatch i, because the pivot is swapped there
    // in the first step of the following swap loop.
    A.unwatch_all();

    ssize_t j = i + (hi-k);

    for (ssize_t s = 0; s < hi-k; ++s) {
        A.swap(i+s, hi-1-s);
        A.mark_swap(i+s, hi-1-s);
    }
    A.unmark_all();

    return std::make_pair(i,j);
}

void QuickSortTernaryLL(SortArray& A, size_t lo, size_t hi)
{
    if (lo + 1 < hi)
    {
        std::pair<ssize_t,ssize_t> mid = PartitionTernaryLL(A, lo, hi);

        QuickSortTernaryLL(A, lo, mid.first);
        QuickSortTernaryLL(A, mid.second, hi);
    }
}

void QuickSortTernaryLL(SortArray& A)
{
    return QuickSortTernaryLL(A, 0, A.size());
}

// ****************************************************************************
// *** Dual-Pivot Quick Sort

// by Sebastian Wild

void dualPivotYaroslavskiy(class SortArray& a, int left, int right)
{
    if (right > left)
    {
        if (a[left] > a[right]) {
            a.swap(left, right);
        }

        const value_type p = a[left];
        const value_type q = a[right];

        a.mark(left);
        a.mark(right);

        volatile ssize_t l = left + 1;
        volatile ssize_t g = right - 1;
        volatile ssize_t k = l;

        a.watch(&l, 3);
        a.watch(&g, 3);
        a.watch(&k, 3);

        while (k <= g)
        {
            if (a[k] < p) {
                a.swap(k, l);
                ++l;
            }
            else if (a[k] >= q) {
                while (a[g] > q && k < g)  --g;
                a.swap(k, g);
                --g;

                if (a[k] < p) {
                    a.swap(k, l);
                    ++l;
                }
            }
            ++k;
        }
        --l;
        ++g;
        a.swap(left, l);
        a.swap(right, g);

        a.unmark_all();
        a.unwatch_all();

        dualPivotYaroslavskiy(a, left, l - 1);
        dualPivotYaroslavskiy(a, l + 1, g - 1);
        dualPivotYaroslavskiy(a, g + 1, right);
    }
}

void QuickSortDualPivot(class SortArray& a)
{
    return dualPivotYaroslavskiy(a, 0, a.size()-1);
}

// ****************************************************************************
// *** Bubble Sort

void BubbleSort(SortArray& A)
{
    for (size_t i = 0; i < A.size()-1; ++i)
    {
        for (size_t j = 0; j < A.size()-1 - i; ++j)
        {
            if (A[j] > A[j + 1])
            {
                A.swap(j, j+1);
            }
        }
    }
}

// ****************************************************************************
// *** Cocktail Shaker Sort

// from http://de.wikibooks.org/wiki/Algorithmen_und_Datenstrukturen_in_C/_Shakersort

void CocktailShakerSort(SortArray& A)
{
    size_t lo = 0, hi = A.size()-1, mov = lo;

    while (lo < hi)
    {
        for (size_t i = hi; i > lo; --i)
        {
            if (A[i-1] > A[i])
            {
                A.swap(i-1, i);
                mov = i;
            }
        }

        lo = mov;

        for (size_t i = lo; i < hi; ++i)
        {
            if (A[i] > A[i+1])
            {
                A.swap(i, i+1);
                mov = i;
            }
        }

        hi = mov;
    }
}

// ****************************************************************************
// *** Gnome Sort

// from http://en.wikipediA.org/wiki/Gnome_sort

void GnomeSort(SortArray& A)
{
    for (size_t i = 1; i < A.size(); )
    {
        if (A[i] >= A[i-1])
        {
            ++i;
        }
        else
        {
            A.swap(i, i-1);
            if (i > 1) --i;
        }
    }
}

// ****************************************************************************
// *** Comb Sort

// from http://en.wikipediA.org/wiki/Comb_sort

void CombSort(SortArray& A)
{
    const double shrink = 1.3;

    bool swapped = false;
    size_t gap = A.size();

    while ((gap > 1) || swapped)
    {
        if (gap > 1) {
            gap = (size_t)((float)gap / shrink);
        }

        swapped = false;

        for (size_t i = 0; gap + i < A.size(); ++i)
        {
            if (A[i] > A[i + gap])
            {
                A.swap(i, i+gap);
                swapped = true;
            }
        }
    }
}

// ****************************************************************************
// *** Odd-Even Sort

// from http://en.wikipediA.org/wiki/Odd%E2%80%93even_sort

void OddEvenSort(SortArray& A)
{
    bool sorted = false;

    while (!sorted)
    {
        sorted = true;

        for (size_t i = 1; i < A.size()-1; i += 2)
        {
            if(A[i] > A[i+1])
            {
                A.swap(i, i+1);
                sorted = false;
            }
        }

        for (size_t i = 0; i < A.size()-1; i += 2)
        {
            if(A[i] > A[i+1])
            {
                A.swap(i, i+1);
                sorted = false;
            }
        }
    }
}

// ****************************************************************************
// *** Shell Sort

// with gaps by Robert Sedgewick from http://www.cs.princeton.edu/~rs/shell/shell.c

void ShellSort(SortArray& A)
{
    size_t incs[16] = { 1391376, 463792, 198768, 86961, 33936,
                        13776, 4592, 1968, 861, 336,
                        112, 48, 21, 7, 3, 1 };

    for (size_t k = 0; k < 16; k++)
    {
        for (size_t h = incs[k], i = h; i < A.size(); i++)
        {
            value_type v = A[i];
            size_t j = i;

            while (j >= h && A[j-h] > v)
            {
                A.set(j, A[j-h]);
                j -= h;
            }

            A.set(j, v);
        }
    }
}

// ****************************************************************************
// *** Heap Sort

// heavily adapted from http://www.codecodex.com/wiki/Heapsort

bool isPowerOfTwo(size_t x)
{
    return ((x != 0) && !(x & (x - 1)));
}

uint32_t prevPowerOfTwo(uint32_t x)
{
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    return x - (x >> 1);
}

int largestPowerOfTwoLessThan(int n)
{
    int k = 1;
    while (k < n) k = k << 1;
    return k >> 1;
}

void HeapSort(SortArray& A)
{
    size_t n = A.size(), i = n / 2;

    // mark heap levels with different colors
    for (size_t j = i; j < n; ++j)
        A.mark(j, log(prevPowerOfTwo(j+1)) / log(2) + 4);

    while (1)
    {
        if (i > 0) {
            // build heap, sift A[i] down the heap
            i--;
        }
        else {
            // pop largest element from heap: swap front to back, and sift
            // front A[0] down the heap
            n--;
            if (n == 0) return;
            A.swap(0,n);

            A.mark(n);
            if (n+1 < A.size()) A.unmark(n+1);
        }

        size_t parent = i;
        size_t child = i*2 + 1;

        // sift operation - push the value of A[i] down the heap
        while (child < n)
        {
            if (child + 1 < n && A[child + 1] > A[child]) {
                child++;
            }
            if (A[child] > A[parent]) {
                A.swap(parent, child);
                parent = child;
                child = parent*2+1;
            }
            else {
                break;
            }
        }

        // mark heap levels with different colors
        A.mark(i, log(prevPowerOfTwo(i+1)) / log(2) + 4);
    }

}

// ****************************************************************************
// *** Radix Sort (counting sort, most significant digit (MSD) first, in-place redistribute)

// by myself (Timo Bingmann)

void RadixSortMSD(SortArray& A, size_t lo, size_t hi, size_t depth)
{
    A.mark(lo); A.mark(hi-1);

    // radix and base calculations
    const unsigned int RADIX = 4;

    unsigned int pmax = floor( log(A.array_max()+1) / log(RADIX) );
    ASSERT(depth <= pmax);

    size_t base = pow(RADIX, pmax - depth);

    // count digits
    std::vector<size_t> count(RADIX, 0);

    for (size_t i = lo; i < hi; ++i)
    {
        size_t r = A[i].get() / base % RADIX;
        ASSERT(r < RADIX);
        count[r]++;
    }

    // inclusive prefix sum
    std::vector<size_t> bkt(RADIX, 0);
    std::partial_sum(count.begin(), count.end(), bkt.begin());

    // mark bucket boundaries
    for (size_t i = 0; i < bkt.size(); ++i) {
        if (bkt[i] == 0) continue;
        A.mark(lo + bkt[i]-1, 3);
    }

    // reorder items in-place by walking cycles
    for (size_t i=0, j; i < (hi-lo); )
    {
        while ( (j = --bkt[ (A[lo+i].get() / base % RADIX) ]) > i )
        {
            A.swap(lo + i, lo + j);
        }
        i += count[ (A[lo+i].get() / base % RADIX) ];
    }

    A.unmark_all();

    // no more depth to sort?
    if (depth+1 > pmax) return;

    // recurse on buckets
    size_t sum = lo;
    for (size_t i = 0; i < RADIX; ++i)
    {
        if (count[i] > 1)
            RadixSortMSD(A, sum, sum+count[i], depth+1);
        sum += count[i];
    }
}

void RadixSortMSD(SortArray& A)
{
    return RadixSortMSD(A, 0, A.size(), 0);
}

// ****************************************************************************
// *** Radix Sort (counting sort, least significant digit (LSD) first, out-of-place redistribute)

// by myself (Timo Bingmann)

void RadixSortLSD(SortArray& A)
{
    // radix and base calculations
    const unsigned int RADIX = 4;

    unsigned int pmax = ceil( log(A.array_max()+1) / log(RADIX) );

    for (unsigned int p = 0; p < pmax; ++p)
    {
        size_t base = pow(RADIX, p);

        // count digits and copy data
        std::vector<size_t> count(RADIX, 0);
        std::vector<value_type> copy(A.size());

        for (size_t i = 0; i < A.size(); ++i)
        {
            size_t r = (copy[i] = A[i]).get() / base % RADIX;
            ASSERT(r < RADIX);
            count[r]++;
        }

        // exclusive prefix sum
        std::vector<size_t> bkt(RADIX+1, 0);
        std::partial_sum(count.begin(), count.end(), bkt.begin()+1);

        // mark bucket boundaries
        for (size_t i = 0; i < bkt.size()-1; ++i) {
            if (bkt[i] >= A.size()) continue;
            A.mark(bkt[i], 3);
        }

        // redistribute items back into array (stable)
        for (size_t i=0; i < A.size(); ++i)
        {
            size_t r = copy[i].get() / base % RADIX;
            A.set( bkt[r]++, copy[i] );
        }

        A.unmark_all();
    }
}

// ****************************************************************************
// *** Use STL Sorts via Iterator Adapters

void StlSort(SortArray& A)
{
    std::sort(MyIterator(&A,0), MyIterator(&A,A.size()));
}

void StlStableSort(SortArray& A)
{
    std::stable_sort(MyIterator(&A,0), MyIterator(&A,A.size()));
}

void StlHeapSort(SortArray& A)
{
    std::make_heap(MyIterator(&A,0), MyIterator(&A,A.size()));
    std::sort_heap(MyIterator(&A,0), MyIterator(&A,A.size()));
}

// ****************************************************************************
// *** BogoSort and more slow sorts

// by myself (Timo Bingmann)

bool BogoCheckSorted(SortArray& A)
{
    size_t i;
    value_type prev = A[0];
    A.mark(0);
    for (i = 1; i < A.size(); ++i)
    {
        value_type val = A[i];
        if (prev > val) break;
        prev = val;
        A.mark(i);
    }

    if (i == A.size()) {
        // this is amazing.
        return true;
    }

    // unmark
    while (i > 0) A.unmark(i--);
    A.unmark(0);

    return false;
}

void BogoSort(SortArray& A)
{
    // keep a permutation of [0,size)
    std::vector<size_t> perm(A.size());

    for (size_t i = 0; i < A.size(); ++i)
        perm[i] = i;

    while (1)
    {
        // check if array is sorted
        if (BogoCheckSorted(A)) break;

        // pick a random permutation of indexes
        std::random_shuffle(perm.begin(), perm.end());

        // permute array in-place
        std::vector<char> pmark(A.size(), 0);

        for (size_t i = 0; i < A.size(); ++i)
        {
            if (pmark[i]) continue;

            // walk a cycle
            size_t j = i;

            //std::cout << "cycle start " << j << " -> " << perm[j] << "\n";

            while ( perm[j] != i )
            {
                ASSERT(!pmark[j]);
                A.swap(j, perm[j]);
                pmark[j] = 1;

                j = perm[j];
                //std::cout << "cycle step " << j << " -> " << perm[j] << "\n";
            }
            //std::cout << "cycle end\n";

            ASSERT(!pmark[j]);
            pmark[j] = 1;
        }

        //std::cout << "permute end\n";

        for (size_t i = 0; i < A.size(); ++i)
            ASSERT(pmark[i]);
    }
}

void BozoSort(SortArray& A)
{
    srand(time(NULL));

    while (1)
    {
        // check if array is sorted
        if (BogoCheckSorted(A)) break;

        // swap two random items
        A.swap(rand() % A.size(), rand() % A.size());
    }
}

// ****************************************************************************
// *** Bitonic Sort

// from http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm

namespace BitonicSortNS {

static const bool ASCENDING = true;    // sorting direction

static void compare(SortArray& A, int i, int j, bool dir)
{
    if (dir == (A[i] > A[j]))
        A.swap(i, j);
}

static void bitonicMerge(SortArray& A, int lo, int n, bool dir)
{
    if (n > 1)
    {
        int m = largestPowerOfTwoLessThan(n);

        for (int i = lo; i < lo + n - m; i++)
            compare(A, i, i+m, dir);

        bitonicMerge(A, lo, m, dir);
        bitonicMerge(A, lo + m, n - m, dir);
    }
}

static void bitonicSort(SortArray& A, int lo, int n, bool dir)
{
    if (n > 1)
    {
        int m = n / 2;
        bitonicSort(A, lo, m, !dir);
        bitonicSort(A, lo + m, n - m, dir);
        bitonicMerge(A, lo, n, dir);
    }
}

} // namespace BitonicSortNS

void BitonicSort(SortArray& A)
{
    BitonicSortNS::bitonicSort(A, 0, A.size(), BitonicSortNS::ASCENDING);
}

// ****************************************************************************
// *** Bitonic Sort as "Parallel" Sorting Network

// from http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm

// modified to first record the recursively generated swap sequence, and then
// sort it back into the order a parallel sorting network would perform the
// swaps in

namespace BitonicSortNetworkNS {

struct swappair_type
{
    // swapped positions
    unsigned int i,j;

    // depth of recursions: sort / merge
    unsigned int sort_depth, merge_depth;

    swappair_type(unsigned int _i, unsigned int _j,
                  unsigned int _sort_depth, unsigned int _merge_depth)
        : i(_i), j(_j),
          sort_depth(_sort_depth), merge_depth(_merge_depth)
    { }

    // order relation for sorting swaps
    bool operator < (const swappair_type& b) const
    {
        if (sort_depth != b.sort_depth)
            return sort_depth > b.sort_depth;

        if (merge_depth != b.merge_depth)
            return merge_depth < b.merge_depth;

        return i < b.i;
    }
};

typedef std::vector<swappair_type> sequence_type;
std::vector<swappair_type> sequence;

void replay(SortArray& A)
{
    for (sequence_type::const_iterator si = sequence.begin();
         si != sequence.end(); ++si)
    {
        if (A[si->i] > A[si->j])
            A.swap(si->i, si->j);
    }
}

static const bool ASCENDING = true; // sorting direction

static void compare(SortArray& /* A */, unsigned int i, unsigned int j, bool dir,
                    unsigned int sort_depth, unsigned int merge_depth)
{
    // if (dir == (A[i] > A[j])) A.swap(i, j);

    if (dir)
        sequence.push_back( swappair_type(i,j, sort_depth, merge_depth) );
    else
        sequence.push_back( swappair_type(j,i, sort_depth, merge_depth) );
}

static void bitonicMerge(SortArray& A, unsigned int lo, unsigned int n, bool dir,
                         unsigned int sort_depth, unsigned int merge_depth)
{
    if (n > 1)
    {
        unsigned int m = largestPowerOfTwoLessThan(n);

        for (unsigned int i = lo; i < lo + n - m; i++)
            compare(A, i, i + m, dir, sort_depth, merge_depth);

        bitonicMerge(A, lo, m, dir, sort_depth, merge_depth+1);
        bitonicMerge(A, lo + m, n - m, dir, sort_depth, merge_depth+1);
    }
}

static void bitonicSort(SortArray& A, unsigned int lo, unsigned int n, bool dir,
                        unsigned int sort_depth)
{
    if (n > 1)
    {
        unsigned int m = n / 2;
        bitonicSort(A, lo, m, !dir, sort_depth+1);
        bitonicSort(A, lo + m, n - m, dir, sort_depth+1);
        bitonicMerge(A, lo, n, dir, sort_depth, 0);
    }
}

void sort(SortArray& A)
{
    sequence.clear();
    bitonicSort(A, 0, A.size(), BitonicSortNS::ASCENDING, 0);
    std::sort(sequence.begin(), sequence.end());
    replay(A);
    sequence.clear();
}

} // namespace BitonicSortNS

void BitonicSortNetwork(SortArray& A)
{
    BitonicSortNetworkNS::sort(A);
}

// ****************************************************************************
// *** Batcher's Odd-Even Merge Sort as "Parallel" Sorting Network

// from http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/oemen.htm

// modified to first record the recursively generated swap sequence, and then
// sort it back into the order a parallel sorting network would perform the
// swaps in

namespace BatcherSortNetworkNS {

struct swappair_type
{
    // swapped positions
    unsigned int i,j;

    // depth of recursions: sort / merge
    unsigned int sort_depth, merge_depth;

    swappair_type(unsigned int _i, unsigned int _j,
                  unsigned int _sort_depth, unsigned int _merge_depth)
        : i(_i), j(_j),
          sort_depth(_sort_depth), merge_depth(_merge_depth)
    { }

    // order relation for sorting swaps
    bool operator < (const swappair_type& b) const
    {
        if (sort_depth != b.sort_depth)
            return sort_depth > b.sort_depth;

        if (merge_depth != b.merge_depth)
            return merge_depth > b.merge_depth;

        return i < b.i;
    }
};

typedef std::vector<swappair_type> sequence_type;
std::vector<swappair_type> sequence;

void replay(SortArray& A)
{
    for (sequence_type::const_iterator si = sequence.begin();
         si != sequence.end(); ++si)
    {
        if (A[si->i] > A[si->j])
            A.swap(si->i, si->j);
    }
}

static void compare(SortArray& A, unsigned int i, unsigned int j,
                    unsigned int sort_depth, unsigned int merge_depth)
{
    // skip all swaps beyond end of array
    ASSERT(i < j);
    if (j >= A.size()) return;

    sequence.push_back( swappair_type(i,j, sort_depth, merge_depth) );

    //if (A[i] > A[j]) A.swap(i, j);
}

// lo is the starting position and n is the length of the piece to be merged, r
// is the distance of the elements to be compared
static void oddEvenMerge(SortArray& A, unsigned int lo, unsigned int n, unsigned int r,
                         unsigned int sort_depth, unsigned int merge_depth)
{
    unsigned int m = r * 2;
    if (m < n)
    {
        // even subsequence
        oddEvenMerge(A, lo, n, m, sort_depth, merge_depth+1);
        // odd subsequence
        oddEvenMerge(A, lo + r, n, m, sort_depth, merge_depth+1);

        for (unsigned int i = lo + r; i + r < lo + n; i += m)
            compare(A, i, i + r, sort_depth, merge_depth);
    }
    else {
        compare(A, lo, lo + r, sort_depth, merge_depth);
    }
}

// sorts a piece of length n of the array starting at position lo
static void oddEvenMergeSort(SortArray& A, unsigned int lo, unsigned int n,
                             unsigned int sort_depth)
{
    if (n > 1)
    {
        unsigned int m = n / 2;
        oddEvenMergeSort(A, lo, m, sort_depth+1);
        oddEvenMergeSort(A, lo + m, m, sort_depth+1);
        oddEvenMerge(A, lo, n, 1, sort_depth, 0);
    }
}

void sort(SortArray& A)
{
    sequence.clear();

    unsigned int n = largestPowerOfTwoLessThan(A.size());
    if (n != A.size()) n *= 2;

    oddEvenMergeSort(A, 0, n, 0);
    std::sort(sequence.begin(), sequence.end());
    replay(A);
    sequence.clear();
}

} // namespace BatcherSortNetworkNS

void BatcherSortNetwork(SortArray& A)
{
    BatcherSortNetworkNS::sort(A);
}

// ****************************************************************************
// *** Smooth Sort

// from http://en.wikipediA.org/wiki/Smoothsort

namespace SmoothSortNS {

static const int LP[] = {
    1, 1, 3, 5, 9, 15, 25, 41, 67, 109,
    177, 287, 465, 753, 1219, 1973, 3193, 5167, 8361, 13529, 21891,
    35421, 57313, 92735, 150049, 242785, 392835, 635621, 1028457,
    1664079, 2692537, 4356617, 7049155, 11405773, 18454929, 29860703,
    48315633, 78176337, 126491971, 204668309, 331160281, 535828591,
    866988873 // the next number is > 31 bits.
};

static void sift(SortArray& A, int pshift, int head)
{
    // we do not use Floyd's improvements to the heapsort sift, because we
    // are not doing what heapsort does - always moving nodes from near
    // the bottom of the tree to the root.

    value_type val = A[head];

    while (pshift > 1)
    {
        int rt = head - 1;
        int lf = head - 1 - LP[pshift - 2];

        if (val.cmp(A[lf]) >= 0 && val.cmp(A[rt]) >= 0)
            break;

        if (A[lf].cmp(A[rt]) >= 0) {
            A.set(head, A[lf]);
            head = lf;
            pshift -= 1;
        }
        else {
            A.set(head, A[rt]);
            head = rt;
            pshift -= 2;
        }
    }

    A.set(head, val);
}

static void trinkle(SortArray& A, int p, int pshift, int head, bool isTrusty)
{
    value_type val = A[head];

    while (p != 1)
    {
        int stepson = head - LP[pshift];

        if (A[stepson].cmp(val) <= 0)
            break; // current node is greater than head. sift.

        // no need to check this if we know the current node is trusty,
        // because we just checked the head (which is val, in the first
        // iteration)
        if (!isTrusty && pshift > 1) {
            int rt = head - 1;
            int lf = head - 1 - LP[pshift - 2];
            if (A[rt].cmp(A[stepson]) >= 0 ||
                A[lf].cmp(A[stepson]) >= 0)
                break;
        }

        A.set(head, A[stepson]);

        head = stepson;
        //int trail = Integer.numberOfTrailingZeros(p & ~1);
        int trail = __builtin_ctz(p & ~1);
        p >>= trail;
        pshift += trail;
        isTrusty = false;
    }

    if (!isTrusty) {
        A.set(head, val);
        sift(A, pshift, head);
    }
}

void sort(SortArray& A, int lo, int hi)
{
    int head = lo; // the offset of the first element of the prefix into m

    // These variables need a little explaining. If our string of heaps
    // is of length 38, then the heaps will be of size 25+9+3+1, which are
    // Leonardo numbers 6, 4, 2, 1.
    // Turning this into a binary number, we get b01010110 = 0x56. We represent
    // this number as a pair of numbers by right-shifting all the zeros and
    // storing the mantissa and exponent as "p" and "pshift".
    // This is handy, because the exponent is the index into L[] giving the
    // size of the rightmost heap, and because we can instantly find out if
    // the rightmost two heaps are consecutive Leonardo numbers by checking
    // (p&3)==3

    int p = 1; // the bitmap of the current standard concatenation >> pshift
    int pshift = 1;

    while (head < hi)
    {
        if ((p & 3) == 3) {
            // Add 1 by merging the first two blocks into a larger one.
            // The next Leonardo number is one bigger.
            sift(A, pshift, head);
            p >>= 2;
            pshift += 2;
        }
        else {
            // adding a new block of length 1
            if (LP[pshift - 1] >= hi - head) {
                // this block is its final size.
                trinkle(A, p, pshift, head, false);
            } else {
                // this block will get merged. Just make it trusty.
                sift(A, pshift, head);
            }

            if (pshift == 1) {
                // LP[1] is being used, so we add use LP[0]
                p <<= 1;
                pshift--;
            } else {
                // shift out to position 1, add LP[1]
                p <<= (pshift - 1);
                pshift = 1;
            }
        }
        p |= 1;
        head++;
    }

    trinkle(A, p, pshift, head, false);

    while (pshift != 1 || p != 1)
    {
        if (pshift <= 1) {
            // block of length 1. No fiddling needed
            //int trail = Integer.numberOfTrailingZeros(p & ~1);
            int trail = __builtin_ctz(p & ~1);
            p >>= trail;
            pshift += trail;
        }
        else {
            p <<= 2;
            p ^= 7;
            pshift -= 2;

            // This block gets broken into three bits. The rightmost bit is a
            // block of length 1. The left hand part is split into two, a block
            // of length LP[pshift+1] and one of LP[pshift].  Both these two
            // are appropriately heapified, but the root nodes are not
            // necessarily in order. We therefore semitrinkle both of them

            trinkle(A, p >> 1, pshift + 1, head - LP[pshift] - 1, true);
            trinkle(A, p, pshift, head - 1, true);
        }

        head--;
    }
}

} // namespace SmoothSortNS

void SmoothSort(SortArray& A)
{
    return SmoothSortNS::sort(A, 0, A.size()-1);
}

// ****************************************************************************
// *** Stooge Sort

void StoogeSort(SortArray& A, int i, int j)
{
    if (A[i] > A[j])
    {
        A.swap(i, j);
    }

    if (j - i + 1 >= 3)
    {
        int t = (j - i + 1) / 3;

        A.mark(i, 3);
        A.mark(j, 3);

        StoogeSort(A, i, j-t);
        StoogeSort(A, i+t, j);
        StoogeSort(A, i, j-t);

        A.unmark(i);
        A.unmark(j);
    }
}

void StoogeSort(SortArray& A)
{
    StoogeSort(A, 0, A.size()-1);
}

// ****************************************************************************
// *** Slow Sort

void SlowSort(SortArray& A, int i, int j)
{
    if (i >= j) return;

    int m = (i + j) / 2;

    SlowSort(A, i, m);
    SlowSort(A, m+1, j);

    if (A[m] > A[j])
        A.swap(m, j);

    A.mark(j, 2);

    SlowSort(A, i, j-1);

    A.unmark(j);
}

void SlowSort(SortArray& A)
{
    SlowSort(A, 0, A.size()-1);
}

// ****************************************************************************
// *** Cycle Sort

// Adapted from http://en.wikipedia.org/wiki/Cycle_sort

void CycleSort(SortArray& array, ssize_t n)
{
    volatile ssize_t cycleStart = 0;
    array.watch(&cycleStart, 16);

    volatile ssize_t rank = 0;
    array.watch(&rank, 3);

    // Loop through the array to find cycles to rotate.
    for (cycleStart = 0; cycleStart < n - 1; ++cycleStart)
    {
        value_type& item = array.get_mutable(cycleStart);

        do {
            // Find where to put the item.
            rank = cycleStart;
            for (ssize_t i = cycleStart + 1; i < n; ++i)
            {
                if (array[i] < item)
                    rank++;
            }

            // If the item is already there, this is a 1-cycle.
            if (rank == cycleStart) {
                array.mark(rank, 2);
                break;
            }

            // Otherwise, put the item after any duplicates.
            while (item == array[rank])
                rank++;

            // Put item into right place and colorize
            std::swap(array.get_mutable(rank), item);
            array.mark(rank, 2);

            // Continue for rest of the cycle.
        }
        while (rank != cycleStart);
    }

    array.unwatch_all();
}

void CycleSort(SortArray& A)
{
    CycleSort(A, A.size());
}
// refer to implemented RadixSortLSD
// cd /home/sound-of-sorting-master/sound-of-sorting-master
void BenRadix(SortArray& inputArray)
{

	std::vector<value_type> copy(inputArray.size());
	const unsigned int RADIX = 4;
	unsigned int p = 0;
	int runTime = 1;

	size_t mod = pow(RADIX, p);

	bool done = false;
	while (!done)
	{
		done = true;
		for (size_t i = 1; i < inputArray.size(); i += 1)
		{
			if (inputArray[i].get() < inputArray[i - 1].get())
			{
				done = false;
			}
		}
		if (done)
		{
			continue;
		}
		if (runTime > 5)
		{
			TimSort(inputArray);
			continue;
		}
		size_t startingNum = 0;
		size_t startingIndex = 0;

		while (startingNum < 4)
		{
			for (size_t i = 0; i < inputArray.size(); i += 1)
			{
				size_t currentDigit = inputArray[i].get() / mod % RADIX;
				if (currentDigit == startingNum)
				{
					copy.at(startingIndex) = inputArray[i];
					startingIndex += 1;
				}
			}
			startingNum += 1;
		}

		mod += 1;

		for (size_t i = 0; i < copy.size(); i += 1)
		{
			inputArray.set(i, copy[i]);
		}

		runTime += 1;
	}


}
void flip(SortArray& arr, size_t i)
{
	size_t start = 0;
	while (start < i)
	{
		value_type temp = arr[start];
		arr.set(start, arr[i]);
		arr.set(i, temp);
		start++;
		i--;
	}
}

/* Returns index of the maximum element in arr[0..n-1] */
size_t findMax(SortArray& arr, size_t n)
{
	size_t mi, i;
	for (mi = 0, i = 0; i < n; ++i)
	{
		if (arr[i] > arr[mi])
		{
			mi = i;
		}

	}
	return mi;
}

// The main function that sorts given array using flip
// operations
void pancakeSort(SortArray& arr)
{
	size_t n = arr.size();
	// Start from the complete array and one by one reduce
	// current size by one
	for (size_t curr_size = n; curr_size > 1; --curr_size)
	{
		// Find index of the maximum element in
		// arr[0..curr_size-1]
		size_t mi = findMax(arr, curr_size);

		// Move the maximum element to end of current array
		// if it's not already at the end
		if (mi != curr_size - 1)
		{
			// To move at the end, first move maximum number
			// to beginning
			flip(arr, mi);

			// Now move the maximum number to end by reversing
			// current array
			flip(arr, curr_size - 1);
		}
	}
}

void inPlaceRadix(SortArray& v) {
	//search for maximum number
	size_t max_number = v[0].get();
	for (size_t i = 1; i<v.size(); i++) {
		if (max_number < v[i].get())
			max_number = v[i].get();
	}

	int bucket[10]; // store first index for that digit
	int bucket_max_index[10]; // store maximum index for that digit
	size_t exp = 1;

	while (max_number / exp > 0) {
		memset(&bucket, 0, sizeof(bucket));
		memset(&bucket_max_index, 0, sizeof(bucket_max_index));
		for (int i = 0; i<v.size(); i++) {
			bucket[(v[i].get() / exp) % 10]++;
		}

		size_t index = v.size();
		for (int i = 9; i >= 0; i--) {
			index -= bucket[i];
			bucket_max_index[i] = index + bucket[i];
			bucket[i] = index;
		}

		index = 0;
		size_t n = v.size();
		std::vector<value_type> temp(n);
		for (size_t i = 0; i<v.size(); i++) {
			size_t digit = (v[i].get() / exp) % 10;
			temp[bucket[digit]] = v[i];
			bucket[digit]++;
		}

		for (size_t i = 0; i<v.size(); i++) {
			v.set(i, temp[i]);
		}

		exp *= 10;
	}
}

void smartSort(SortArray& a)
{
	int runTime = 1;
	while (runTime <= 3)
	{
		int left = 0;
		int right = a.size() - 1;
		if (a[left] > a[right]) {
			a.swap(left, right);
		}

		const value_type p = a[left];
		const value_type q = a[right];

		a.mark(left);
		a.mark(right);

		volatile ssize_t l = left + 1;
		volatile ssize_t g = right - 1;
		volatile ssize_t k = l;

		a.watch(&l, 3);
		a.watch(&g, 3);
		a.watch(&k, 3);

		while (k <= g)
		{
			if (a[k] < p) {
				a.swap(k, l);
				++l;
			}
			else if (a[k] >= q) {
				while (a[g] > q && k < g)  --g;
				a.swap(k, g);
				--g;

				if (a[k] < p) {
					a.swap(k, l);
					++l;
				}
			}
			++k;
		}
		--l;
		++g;
		a.swap(left, l);
		a.swap(right, g);

		a.unmark_all();
		a.unwatch_all();

		int oldleft = left;
		int oldright = right;

		volatile ssize_t oldl = oldleft + 1;
		volatile ssize_t oldg = oldright - 1;

		// all 3 iterations
		switch (runTime % 3)
		{
		case 1:
		{
			right = oldl - 1;
			left = oldleft;
		}
		case 2:
		{
			left = oldl + 1;
			right = oldg - 1;
		}
		case 0:
		{
			left = oldg + 1;
			right = oldright;
		}
		}

		runTime += 1;
	}

	// Timsort the mostly sorted list
	TimSort(a);
}

void CocktailShakerSort2(SortArray& A, size_t lo, size_t hi)
{
	size_t mov = lo;

	while (lo < hi)
	{
		for (size_t i = hi; i > lo; --i)
		{
			if (A[i - 1] > A[i])
			{
				A.swap(i - 1, i);
				mov = i;
			}
		}

		lo = mov;

		for (size_t i = lo; i < hi; ++i)
		{
			if (A[i] > A[i + 1])
			{
				A.swap(i, i + 1);
				mov = i;
			}
		}

		hi = mov;
	}
}

void dualCocktailMerge(SortArray& A)
{
	// split the array
	size_t mid = (size_t)(A.size() / 2);
	size_t firstQuarter = (size_t)(mid / 2);
	size_t thirdQuarter = mid + firstQuarter;

	A.mark(firstQuarter, 2);
	A.mark(mid, 2);
	A.mark(thirdQuarter, 2);

	// cocktail each quarter
	CocktailShakerSort2(A, 0, firstQuarter);
	A.unmark(firstQuarter);
	CocktailShakerSort2(A, firstQuarter, mid);
	A.unmark(mid);
	CocktailShakerSort2(A, mid, thirdQuarter);
	A.unmark(thirdQuarter);
	CocktailShakerSort2(A, thirdQuarter, A.size() - 1);

	// merge the quarters (timsort is godlike)
	TimSort(A);
}

void pigeonholeSort(SortArray& arr) // TIMING NOT RELIABLE - WORK DONE IN COPY ARRAY
{
	size_t n = arr.size();
	// Find minimum and maximum values in arr[]
	int min = arr[0].get(), max = arr[0].get();
	for (size_t i = 1; i < n; i++)
	{
		if (arr[i].get() < min)
			min = arr[i].get();
		if (arr[i].get() > max)
			max = arr[i].get();
	}
	int range = max - min + 1; // Find range

							   // Create an array of vectors. Size of array
							   // range. Each vector represents a hole that
							   // is going to contain matching elements.
	std::vector<value_type> holes[range];

	// Traverse through input array and put every
	// element in its respective hole
	for (size_t i = 0; i < n; i++)
	{
		holes[arr[i].get() - min].push_back(arr[i]);
		arr.mark(i, 6);
	}

	// Traverse through all holes one by one. For
	// every hole, take its elements and put in
	// array.
	size_t index = 0;  // index in sorted array
	for (size_t i = 0; i < range; i++)
	{
		std::vector<value_type>::iterator it;
		for (it = holes[i].begin(); it != holes[i].end(); ++it)
			arr.set(index++, *it);
	}
}

void InsertionSortIndex(SortArray& A, size_t length)
{
	for (size_t i = 1; i < length; ++i)
	{
		value_type key = A[i];
		A.mark(i);

		ssize_t j = i - 1;
		while (j >= 0 && A[j] > key)
		{
			A.swap(j, j + 1);
			j--;
		}

		A.unmark(i);
	}
}

void flashsort(SortArray& array, size_t length)
{
	if (length == 0) return;

	//20% of the number of elements or 0.2n classes will
	//be used to distribute the input data set into
	//there must be at least 2 classes (hence the addition)
	int m = (int)((0.2 * length) + 2);

	//-------CLASS FORMATION-------

	//O(n)
	//compute the max and min values of the input data
	int min, max;
	size_t maxIndex;
	min = max = array[0].get();
	maxIndex = 0;

	for (size_t i = 1; i < length - 1; i += 2)
	{
		int small;
		int big;
		size_t bigIndex;

		//which is bigger A(i) or A(i+1)
		if (array[i] < array[i + 1])
		{
			small = array[i].get();
			big = array[i + 1].get();
			bigIndex = i + 1;
		}
		else
		{
			big = array[i].get();
			bigIndex = i;
			small = array[i + 1].get();
		}

		if (big > max)
		{
			max = big;
			maxIndex = bigIndex;
		}

		if (small < min)
		{
			min = small;
		}
	}

	//do the last element
	if (array[length - 1].get() < min)
	{
		min = array[length - 1].get();
	}
	else if (array[length - 1].get() > max)
	{
		max = array[length - 1].get();
		maxIndex = length - 1;
	}

	if (max == min)
	{
		//all the elements are the same
		return;
	}

	//dynamically allocate the storage for L
	//note that L is in the range 1...m (hence
	//the extra 1)
	size_t* L = new size_t[m + 1];

	//O(m)
	//initialize L to contain all zeros (L[0] is unused)
	for (size_t t = 1; t <= m; t++)
	{
		L[t] = 0;
	}

	//O(n)
	//use the function K(A(i)) = 1 + INT((m-1)(A(i)-Amin)/(Amax-Amin))
	//to classify each A(i) into a number from 1...m
	//(note that this is mainly just a percentage calculation)
	//and then store a count of each distinct class K in L(K)
	//For instance, if there are 22 A(i) values that fall into class
	//K == 5 then the count in L(5) would be 22

	//IMPORTANT: note that the class K == m only has elements equal to Amax

	//precomputed constant
	double c = (m - 1.0) / (max - min);
	size_t K;
	for (size_t h = 0; h < length; h++)
	{
		//classify the A(i) value
		K = ((size_t)((array[h].get() - min) * c)) + 1;

		//assert: K is in the range 1...m

		//add one to the count for this class
		L[K] += 1;
	}

	//O(m)
	//sum over each L(i) such that each L(i) contains
	//the number of A(i) values that are in the ith
	//class or lower (see counting sort for more details)
	for (K = 2; K <= m; K++)
	{
		L[K] = L[K] + L[K - 1];
	}

	//-------PERMUTATION-------

	//swap the max value with the first value in the array
	value_type temp = array[maxIndex];
	array.set(maxIndex, array[0]);
	array.set(0, temp);

	//Except when being iterated upwards,
	//j always points to the first A(i) that starts
	//a new class boundary && that class hasn't yet
	//had all of its elements moved inside its borders;

	//This is called a cycle leader since you know
	//that you can begin permuting again here. You know
	//this becuase it is the lowest index of the class
	//and as such A(j) must be out of place or else all
	//the elements of this class have already been placed
	//within the borders of the this class (which means
	//j wouldn't be pointing to this A(i) in the first place)
	int j = 0;

	//K is the class of an A(i) value. It is always in the range 1..m
	K = m;

	//the number of elements that have been moved
	//into their correct class
	size_t numMoves = 0;

	//O(n)
	//permute elements into their correct class; each
	//time the class that j is pointing to fills up
	//then iterate j to the next cycle leader
	//
	//do not use the n - 1 optimization because that last element
	//will not have its count decreased (this causes trouble with
	//determining the correct classSize in the last step)
	while (numMoves < length)
	{
		//if j does not point to the begining of a class
		//that has at least 1 element still needing to be
		//moved to within the borders of the class then iterate
		//j upward until such a class is found (such a class
		//must exist). In other words, find the next cycle leader
		while (j >= L[K])
		{
			j++;
			//classify the A(j) value
			K = ((int)((array[j].get() - min) * c)) + 1;
		}

		//evicted always holds the value of an element whose location
		//in the array is free to be written into //aka FLASH
		int evicted = array[j].get();
		value_type evicted2 = array[j];

		//while j continues to meet the condition that it is
		//pointing to the start of a class that has at least one
		//element still outside its borders (the class isn't full)
		while (j < L[K])
		{
			//compute the class of the evicted value
			K = ((int)((evicted - min) * c)) + 1;

			//get a location that is inside the evicted
			//element's class boundaries
			size_t location = L[K] - 1;

			//swap the value currently residing at the new
			//location with the evicted value
			value_type temp = array[location];
			array.set(location, evicted2);
			evicted2 = temp;
			evicted = temp.get();

			//decrease the count for this class
			//see counting sort for why this is done
			L[K] -= 1;

			//another element was moved
			numMoves++;
		}
	}

	//-------RECURSION or STRAIGHT INSERTION-------

	//if the classes do not have the A(i) values uniformly distributed
	//into each of them then insertion sort will not produce O(n) results;

	//look for classes that have too many elements; ideally each class
	//(except the topmost or K == m class) should have about n/m elements;
	//look for classes that exceed n/m elements by some threshold AND have
	//more than some minimum number of elements to flashsort recursively

	//if the class has 25% more elements than it should
	int threshold = (int)(1.25 * ((length / m) + 1));
	const int minElements = 30;

	//for each class decide whether to insertion sort its members
	//or recursively flashsort its members;
	//skip the K == m class because it is already sorted
	//since all of the elements have the same value
	for (K = m - 1; K >= 1; K--)
	{
		//determine the number of elments in the Kth class
		size_t classSize = L[K + 1] - L[K];

		//if the class size is larger than expected but not
		//so small that insertion sort could make quick work
		//of it then...
		if (classSize > threshold && classSize > minElements)
		{
			//...attempt to flashsort the class. This will work
			//well if the elements inside the class are uniformly
			//distributed throughout the class otherwise it will
			//perform badly, O(n^2) worst case, since we will have
			//performed another classification and permutation step
			//and not succeeded in making the problem significantly
			//smaller for the next level of recursion. However,
			//progress is assured since at each level the elements
			//with the maximum value will get sorted.
			flashsort(array, classSize);
		}
		else //perform insertion sort on the class
		{
			if (classSize > 1)
			{
				InsertionSortIndex(array, array.size());
				return;
			}
		}
	}

	delete[] L;
}

void flashSortMain(SortArray& array)
{
	flashsort(array, array.size());
}

void gravitySort(SortArray& A) // TIMING NOT RELIABLE - WORK DONE IN COPY ARRAY
{
	// make copy array of ints and ArrayItems and get max

	int copy[A.size()];
	std::vector<value_type> copy2(A.size());

	size_t max = 0;

	for (size_t i = 0; i < A.size(); i += 1)
	{
		copy[i] = A[i].get();
		copy2[i] = A[i];
		if (copy[i] > copy[max])
			max = i;
	}

	size_t index = 0;

	// iterate through array, let each int fall one value and check if it hits the floor
	while (copy[max] > 0)
	{
		for (size_t o = 0; o < A.size(); o += 1)
		{
			copy[o] -= 1;
			if (copy[o] == 0)
			{
				A.set(index, copy2[o]);
				A.mark(o, 6);
				index += 1;
			}
		}
	}
	// set A to finished array (not needed because it's in place now)
	//for (size_t i = 0; i < A.size(); i += 1)
	//{
	//	A.set(i, copy2[i]);
	//}
}

void arl(SortArray& arr, int start, int end, int leftBitIndex)
{
	int numElements = end - start + 1;

	if (numElements == 0)
	{
		return;
	}

	const int nbmax = 11;
	const int nbmin = 4;

	//compute the number of bits to be used in the next radix digit
	//the digit must be less than nbmax bits wide; note however that
	//it can be smaller than nbmin bits
	int numBits = leftBitIndex + 1;
	if (numBits > 11)
	{
		numBits = 11;
	}

	while ((1 << numBits) > numElements && numBits > nbmin)
	{
		numBits--;
	}

	//assert: numBits will be at least 1
	unsigned int digitRange = 0x00000001 << numBits;

	//allocate the two supporting arrays - I don't think two arrays
	//are strictly necessary (see my flashsort imp.) but I use them
	//to be consistant with the algorithm as it is described
	int* equalEndIndex = new int[digitRange];
	int* equalStartIndex = new int[digitRange];

	//init counts to zero
	for (unsigned int i = 0; i < digitRange; i++)
	{
		equalEndIndex[i] = 0;
	}

	unsigned int value;
	unsigned int offset = leftBitIndex + 1 - numBits;
	unsigned int mask;

	//create the new radix digit mask
	const unsigned int allones = 0xFFFFFFFF;
	unsigned int shiftLeft = allones << offset;
	unsigned int shiftRight = allones >> (31 - leftBitIndex);
	mask = shiftLeft & shiftRight;

	//keep a count for each of the digitRange different values
	//of the new radix digit
	for (int j = start; j <= end; j++)
	{
		//determine the value for the digit of interest
		value = (arr[j].get() & mask) >> offset;

		//add one to the count for this value
		equalEndIndex[value] += 1;
	}

	//holds the ending index + 1 and the starting index respectively
	//for each of the 2^numBits distinct values of the new radix digit
	//since each radix digit will, in general, have several elements
	//that have equal valued radix digits; the starting index array
	//will be decremented until it reaches the actual starting index
	//after the permutation step below;
	//see counting sort for more details on this summation
	equalStartIndex[0] = equalEndIndex[0];
	for (unsigned int m = 1; m < digitRange; m++)
	{
		equalEndIndex[m] = equalEndIndex[m] + equalEndIndex[m - 1];
		equalStartIndex[m] = equalEndIndex[m];
	}

	//******* permutation step ********
	//see flashsort for an detailed explanation

	//cycle leader
	int leader = start;
	int numMoves = 0;

	//an index into the equalStartIndex array
	unsigned int starts_i = digitRange - 1;

	//while there are still more elements to permute into place
	while (numMoves < numElements)
	{
		//find the next cycle leader
		while ((leader - start) >= equalStartIndex[starts_i])
		{
			leader++;
			starts_i = (arr[leader].get() & mask) >> offset;
		}

		int evicted = arr[leader].get();

		//permute elements until a new cycle leader is needed
		while ((leader - start) < equalStartIndex[starts_i])
		{
			starts_i = (evicted & mask) >> offset;

			//the new location of the evicted element
			int location = equalStartIndex[starts_i] - 1 + start;

			//swap the value currently residing at the new
			//location with the evicted value
			int temp = arr[location].get();
			arr.set(location, ArrayItem(evicted));
			evicted = temp;

			//see counting sort
			equalStartIndex[starts_i] -= 1;

			//another element was moved
			numMoves++;
		}
	}

	//******* recursive step ********

	const int SEGMENT_SIZE_THRESHOLD = 20;

	//if there are still more digits to the right to sort on
	if (leftBitIndex + 1 - numBits > 0) //correction to the algorithm as stated in Maus's paper
	{
		//for all the equal valued elements discovered for this digit
		for (unsigned int s = 0; s < digitRange; s++)
		{
			int segmentSize = equalEndIndex[s] - equalStartIndex[s];
			if (segmentSize > 1)
			{
				if (segmentSize > SEGMENT_SIZE_THRESHOLD)
				{
					arl(arr, start + equalStartIndex[s], //start index for segment
						start + equalEndIndex[s] - 1, //end index for segment
						leftBitIndex - numBits);
				}
				else
				{
					InsertionSortIndex(arr, numElements);
                    return;
				}
			}
		}
	}

	delete[] equalStartIndex;
	delete[] equalEndIndex;
}


void adaptiveRadixLeft(SortArray &a)
{
	int p[a.size()];

	for (int i = 0; i < a.size(); i++)
	{
		p[i] = a[i].get();
	}

	unsigned int xxor = p[0];
	for (int i = 1; i < a.size(); i++)
	{
		xxor |= p[i];
	}

	//find the most significant bit that is set
	unsigned int mask = 0x80000000;
	int index = -1;
	for (int j = 31; j >= 0; j--)
	{
		unsigned int value = xxor & mask;
		value = value >> j;
		mask = mask >> 1;
		if (value)
		{
			index = j;
			break;
		}
	}

	if (index == -1)
	{
		//all the values are 0
		return;
	}

	for (int i = 0; i < a.size(); i++)
    {
        a.set(i, ArrayItem(p[i]));
    }

	arl(a, 0, a.size() - 1, index);
}

void threeWayPartition(SortArray& arr, int left, int right, int &i, int &j)
{
	i = left - 1;
	j = right;
	volatile ssize_t p = left - 1;
	volatile ssize_t q = right;
	int v = arr[right].get();

  arr.mark(left);
  arr.mark(right);

  arr.watch(&p, 3);
  arr.watch(&q, 3);

	while (true)
	{
		while (arr[++i].get() < v);

		while (v < arr[--j].get())
		{
			if (j == left)
			{
				break;
			}
		}

		if (i >= j)
		{
			break;
		}

    arr.swap(i, j);

		if (arr[i].get() == v)
		{
			p++;

      arr.swap(i, p);
		}

		if (arr[j].get() == v)
		{
			q--;

      arr.swap(j, q);
		}
	}

  arr.swap(i, right);

	j = i - 1;
	for (int k = left; k < p; k++, j--)
	{
    arr.swap(j, k);
	}

	i = i + 1;

	for (int k = right - 1; k > q; k--, i++)
	{
    arr.swap(i, k);
	}

  arr.unmark_all();
  arr.unwatch_all();
}
void threeWayQuicksort(SortArray& arr, int left, int right)
{
	if (right <= left)
	{
		return;
	}

	int i;
	int j;

	threeWayPartition(arr, left, right, i, j);

	threeWayQuicksort(arr, left, j);
	threeWayQuicksort(arr, i, right);
}

void threeWayQuicksortMain(SortArray& arr)
{
	threeWayQuicksort(arr, 0, arr.size() - 1);
}
