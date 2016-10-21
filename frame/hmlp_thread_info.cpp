#include <hmlp_thread_info.hpp>

namespace hmlp
{

worker::worker( thread_communicator *comm ) :
  tid( 0 ), jc_id( 0 ), pc_id( 0 ), ic_id( 0 ), jr_id( 0 )
{
  int tmp;

  tid   = omp_get_thread_num();
  tmp   = tid;

  my_comm = comm;

  jc_id = tmp / ( my_comm->GetNumThreads() / my_comm->GetNumGroups() );
  tmp   = tmp % ( my_comm->GetNumThreads() / my_comm->GetNumGroups() );

  jc_comm = &(my_comm->kids[ jc_id ]);

  pc_id = tmp / ( jc_comm->GetNumThreads() / jc_comm->GetNumGroups() );
  tmp   = tmp % ( jc_comm->GetNumThreads() / jc_comm->GetNumGroups() );

  pc_comm = &(jc_comm->kids[ pc_id ]);

  ic_id = tmp / ( pc_comm->GetNumThreads() / pc_comm->GetNumGroups() );
  jr_id = tmp % ( pc_comm->GetNumThreads() / pc_comm->GetNumGroups() );

  ic_comm = &(pc_comm->kids[ ic_id ]);
};

};

