/*
 * University of Illinois Open Source License
 * Copyright 2010 Luthey-Schulten Group,
 * All rights reserved.
 *
 * Developed by: Luthey-Schulten Group
 *               University of Illinois at Urbana-Champaign
 *               http://www.scs.uiuc.edu/~schulten
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the Software), to deal with
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to
 * do so, subject to the following conditions:
 *
 * - Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimers.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimers in the documentation
 * and/or other materials provided with the distribution.
 *
 * - Neither the names of the Luthey-Schulten Group, University of Illinois at
 * Urbana-Champaign, nor the names of its contributors may be used to endorse or
 * promote products derived from this Software without specific prior written
 * permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS WITH THE SOFTWARE.
 *
 * Author(s): Elijah Roberts
 */
#ifndef TIMINGCONSTANTS_H_
#define TIMINGCONSTANTS_H_

#define PROF_MAX_THREADS							1
#define PROF_MAX_EVENTS								9999
#define MACOSX
#define PROF_OUT_FILE								/Users/tel/git/stocuda/stocuda.eprof

#define PROF_THREAD_VARIABLE_START                  3

#define PROF_MAIN_RUN                               1
#define PROF_SIM_RUN                                2

#define PROF_MASTER_READ_STATIC_MSG                 3
#define PROF_MASTER_READ_FINISHED_MSG               4
#define PROF_MASTER_FINISHED_THREAD                 5
#define PROF_MASTER_SLEEP                           6

#define PROF_REPLICATE_EXECUTE                      10
#define PROF_REPLICATE_WRITE_DATASET                11

#define PROF_INIT_MARKING							100
#define PROF_CUDAMALLOC								101
#define PROF_CUDAMEMCOPY_TO							102
#define PROF_UPDATE_KERNEL							103
#define PROF_CUDAMEMCOPY_FROM						104
#define PROF_UPDATEM								105

#endif /* TIMINGCONSTANTS_H_ */
