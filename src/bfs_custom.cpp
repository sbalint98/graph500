//Stub for custom BFS implementations

#include "common.h"
#include "aml.h"
#include "csr_reference.h"
#include "bitmap_reference.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <stdint.h>
#include <cassert>
#include <atomic>
#include <sycl/sycl.hpp>

//VISITED bitmap parameters
unsigned long *visited;
int64_t visited_size;

int64_t *pred_glob;
extern int64_t *column;
int *rowstarts;
oned_csr_graph g;

int select_tbb_dev(const sycl::device& dev){
    if (dev.get_platform() == hipsycl::rt::backend_id::tbb){
        return 1;
    }else{
        return -1;
    }
}

class tbb_selector {
public:
  int operator()(const sycl::device &dev) const { return select_tbb_dev(dev); }
};

int64_t
int64_cas(int64_t* p, int64_t oldval, int64_t newval)
{
  return __sync_bool_compare_and_swap (p, oldval, newval);
}

int64_t
int64_fetch_add (int64_t* p, int64_t incr)
{
  return __sync_fetch_and_add (p, incr);
}

sycl::device tbb_dev = sycl::device(tbb_selector());
sycl::queue queue(tbb_dev);

//user should provide this function which would be called once to do kernel 1: graph convert
void make_graph_data_structure(const tuple_graph* const tg) {
	//graph conversion, can be changed by user by replacing oned_csr.{c,h} with new graph format 
	convert_graph_to_oned_csr(tg, &g);

	column=g.column;
	visited_size = (g.nlocalverts + ulong_bits - 1) / ulong_bits;
	visited = reinterpret_cast<unsigned long*>(xmalloc(visited_size*sizeof(unsigned long)));
	//user code to allocate other buffers for bfs
}

//user should provide this function which would be called several times to do kernel 2: breadth first search
//pred[] should be root for root, -1 for unrechable vertices
//prior to calling run_bfs pred is set to -1 by calling clean_pred
#define THREAD_BUF_LEN 16384
void run_bfs(int64_t root, int64_t* pred) {
	pred_glob = pred;
	// for(int i = 0; i < g.nglobalverts+1; i++){
	// 	std::cout << g.rowstarts[i] << " ";
	// }
	// std::cout << std::endl;
	int64_t *bfs_tree = pred;

	int64_t * vlist = NULL;
	int64_t k1;
	int64_t k2;
	int64_t* k2_ptr = &k2;

	int64_t nv = g.nglobalverts;
	int64_t srcvtx = root;

	//*max_vtx_out = maxvtx;
	
	vlist = reinterpret_cast<int64_t*>(xmalloc(g.nglobalverts * sizeof (*vlist)));
	if (!vlist){assert(false);}

	for (k1 = 0; k1 < nv; ++k1){
		pred[k1] = -1;
	}

	vlist[0] = srcvtx;
	bfs_tree[srcvtx] = srcvtx;
	k1 = 0; k2 = 1;	
	while (k1 != k2) {
		const int64_t oldk2 = k2;
		queue.submit([&](sycl::handler& cgh){
			cgh.single_task([=](){
				int64_t kbuf = 0;
				int64_t nbuf[THREAD_BUF_LEN];
				for (int64_t k = k1; k < oldk2; ++k) {
					const int64_t v = vlist[k];
					const int64_t veo = g.rowstarts[v+1];
					for (int64_t vo = g.rowstarts[v]; vo < veo; ++vo) {
						const int64_t j = g.column[vo];
						if (bfs_tree[j] == -1) {
							if (int64_cas (&bfs_tree[j], -1, v)) {
								if (kbuf < THREAD_BUF_LEN) {
									nbuf[kbuf++] = j;
								}else {
									int64_t voff = int64_fetch_add (k2_ptr, THREAD_BUF_LEN), vk;
									assert (voff + THREAD_BUF_LEN <= nv);
									for (vk = 0; vk < THREAD_BUF_LEN; ++vk){
										vlist[voff + vk] = nbuf[vk];
									}
									nbuf[0] = j;
									kbuf = 1;
								}
							}
						}
					}
				}
				if (kbuf){
					int64_t voff = int64_fetch_add (k2_ptr, kbuf), vk;
					assert (voff + kbuf <= nv);
					for (vk = 0; vk < kbuf; ++vk){
						vlist[voff + vk] = nbuf[vk];
					}
				}
			});
		});
		k1 = oldk2;
		queue.wait();
	}
	free(vlist);
}

//we need edge count to calculate teps. Validation will check if this count is correct
//user should change this function if another format (not standart CRS) used
void get_edge_count_for_teps(int64_t* edge_visit_count) {
	long i,j;
	long edge_count=0;
	for(i=0;i<g.nlocalverts;i++)
		if(pred_glob[i]!=-1) {
			for(j=g.rowstarts[i];j<g.rowstarts[i+1];j++)
				if(COLUMN(j)<=VERTEX_TO_GLOBAL(my_pe(),i))
					edge_count++;
		}
	aml_long_allsum(&edge_count);
	*edge_visit_count=edge_count;
}

//user provided function to initialize predecessor array to whatevere value user needs
void clean_pred(int64_t* pred) {
	int i;
	for(i=0;i<g.nlocalverts;i++) pred[i]=-1;
}

//user provided function to be called once graph is no longer needed
void free_graph_data_structure(void) {
	free_oned_csr_graph(&g);
	free(visited);
}

//user should change is function if distribution(and counts) of vertices is changed
size_t get_nlocalverts_for_pred(void) {
	return g.nlocalverts;
}
