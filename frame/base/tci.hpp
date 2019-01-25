#ifndef HMLP_TPI_HPP
#define HMLP_TPI_HPP

#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <tuple>

#include <omp.h>

#include <base/util.hpp>

using namespace std;


namespace hmlp
{
typedef enum 
{
  HMLP_SCHEDULE_DEFAULT,
  HMLP_SCHEDULE_ROUND_ROBIN,
  HMLP_SCHEDULE_UNIFORM,
  HMLP_SCHEDULE_HEFT
} SchedulePolicy;


class Range
{
  public:

    Range( int beg, int end, int inc );

    int beg();

    int end();

    int inc();

    void Print( int prefix );

  private:

    tuple<int, int, int > info;

};


/** @brief Wrapper for omp or pthread mutex.  */ 
class Lock
{
  public:

    Lock();

    ~Lock();

    hmlpError_t Acquire();

    hmlpError_t Release();

  private:
#ifdef USE_PTHREAD_RUNTIME
    pthread_mutex_t lock;
#else
    omp_lock_t lock;
#endif
}; /** end class Lock */

  
namespace tci
{

class Context
{
  public:

    void* buffer = NULL;

	  volatile bool barrier_sense = false;

    volatile int barrier_threads_arrived = 0;

    void Barrier( int size );

}; /** end class Context */


class Comm
{
  public:

    Comm();
    
    Comm( Context* context );
    
    Comm( Comm* parent, Context* context, int assigned_size, int assigned_rank );

    Comm Split( int num_groups );

    bool Master();

    void Barrier();

    void Send( void** sent_object );

    void Recv( void** recv_object );

    template<typename Arg>
    void Bcast( Arg& buffer, int root )
    {
      if ( rank == root ) Send( (void**)&buffer );
      Barrier();
      if ( rank != root ) Recv( (void**)&buffer );
    };

    template<int ALIGN_SIZE, typename T>
    T *AllocateSharedMemory( size_t count )
    {
      T* ptr = NULL;
      if ( Master() ) ptr = hmlp_malloc<ALIGN_SIZE, T>( count );
      Bcast( ptr, 0 );
      return ptr;
    };

    template<typename T>
    void FreeSharedMemory( T *ptr )
    {
      Barrier();
      if ( Master() ) hmlp_free( ptr );
    };

    void Create1DLocks( int n );

    void Destroy1DLocks();

    void Create2DLocks( int m, int n );

    void Destroy2DLocks();

    void Acquire1DLocks( int i );

    void Release1DLocks( int i );

    void Acquire2DLocks( int i , int j );

    void Release2DLocks( int i , int j );

    int GetCommSize();

    int GetCommRank();

    int GetGangSize();

    int GetGangRank();

    int BalanceOver1DGangs( int n, int default_size, int nb );

    Range DistributeOver1DThreads( int beg, int end, int nb );

    Range DistributeOver1DGangs( int beg, int end, int nb );

    void Print( int prefix );

    Comm* parent = NULL;

  private:

    string name;

    int rank = 0;

    int size = 1;

    int gang_rank = 0;

    int gang_size = 1;

    Context* context = NULL;
   
    vector<hmlp::Lock>* lock1d = NULL;

    vector<vector<hmlp::Lock>>* lock2d = NULL;

}; /** end class Comm */

template<typename FUNC, typename... Args>
void Parallelize( tci::Comm* comm, FUNC func, Args&&... args )
{
  if ( comm )
  {
    func( *comm, args... );
  }
  else
  {
    /** Create a shared context pointer for communication. */
    Context context;
    /** Create a parallel section with omp_get_num_threads(). */
    #pragma omp parallel 
    {
      /** Create a global communicator with the shared context. */
      Comm CommGLB( &context );
      /** Now call the function in parallel. */
      func( CommGLB, args... );
    } /** end pragma omp parallel */
  }
}; /** end Parallelize() */

}; /** end namespace tci */
}; /** end namespace hmlp */
#endif
