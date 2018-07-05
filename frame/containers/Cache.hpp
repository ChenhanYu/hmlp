#ifndef CACHE_HPP
#define CACHE_HPP


#include <unordered_map>

#include <hmlp_runtime.hpp>
#include <containers/data.hpp>

using namespace std;
using namespace hmlp;


namespace hmlp
{

/**
 *  @brief Cache1D<NSET, NWAY, T> creates a layer of cache
 *         with NSET that directly maps a 1D array. 
 *         The direct map is [ id % NSET ]. Each set has
 *         NWAY that are fully associated. 
 *
 */ 
template<size_t NSET, size_t NWAY, typename T>
class Cache1D
{
	public:

		Cache1D( size_t unit )
		{
			/** resize table and locks */
		  table.resize( NSET );
			locks.resize( NSET );

			/** resize frequency */
			freq.resize( NSET * NWAY , 0 );

			/** resize buffer */
			buffer.resize( unit, NSET * NWAY );

			/** all units */
			all_units.resize( unit );
			for ( size_t i = 0; i < unit; i ++ ) all_units[ i ] = i;

		};

	  Data<T> Read( size_t id )
		{
			Data<T> ret;

			/** use direct map to find the set */
			auto &set = table[ id % NSET ];

			locks[ id % NSET ].Acquire();
			{
				/** use std::map::find inside the fully associated set */
				auto it = set.find( id );
				if ( it != set.end() ) 
				{
					size_t line_id = (*it).second;
					vector<size_t> J( 1, line_id );
					/** cache hit */
					ret = buffer( all_units, J );
					/** increate frequency */
					freq[ line_id ] ++;
				}
			}
			locks[ id % NSET ].Release();

			return ret;
		};

		void Write( size_t id,  Data<T> &input )
		{
			/** use direct map to find the set */
			auto &set = table[ id % NSET ];

			/** compute the offset of the whole NSET-by-NWAY cachline */
		  size_t offset = ( id % NSET ) * NWAY;

			locks[ id % NSET ].Acquire();
			{
				/** early return if id has been written to the cache */
				if ( set.find( id ) != set.end() )
				{
			    locks[ id % NSET ].Release();
          return;
				}

				/** column id of the buffer */
				size_t line_id = 0;

				if ( set.size() < NWAY )
				{
					/** insert id into table[ id % NSET ] */
					line_id = offset + set.size();
				}
				else
				{
					assert( set.size() == NWAY );

					/** cache replacement */
					size_t min_freq = freq[ offset ];
					/** compute the minimal frequency */
					for ( size_t i = 0; i < NWAY; i ++ )
						if ( freq[ offset + i ] < min_freq ) min_freq = freq[ offset + i ];
					/** match the cahceline and replace */
					size_t id_to_erase = 0;
					//size_t line_id = 0;
					for ( auto it = set.begin(); it != set.end(); it ++ )
					{
						if ( freq[ (*it).second ] == min_freq )
						{
							id_to_erase = (*it).first;
							line_id = (*it).second;
						}
					}
					set.erase( id_to_erase );
					//set[ id ] = line_id;
					//for ( size_t i = 0; i < buffer.row(); i ++ ) 
					//	buffer( i, line_id ) = input[ i ];
				}

				set[ id ] = line_id;
				for ( size_t i = 0; i < buffer.row(); i ++ ) 
					buffer( i, line_id ) = input[ i ];

			}
			locks[ id % NSET ].Release();
		};


	private:


		vector<size_t> all_units;

		vector<Lock> locks;

		vector<unordered_map<size_t, size_t>> table;

		vector<size_t> freq;

		Data<T> buffer;

};


template<size_t NSET, size_t NWAY, typename T>
class Cache2D
{
	public:

		Cache2D( size_t n )
		{
			this->n = n;
			this->ldway = ( n / NSET ) + 1;
			locks.resize( NSET );
      table.resize( NSET );
			for ( size_t i = 0; i < NSET; i ++ ) 
			{
				locks[ i ].resize( NSET );
        table[ i ].resize( NSET );
			}
		};

		Data<T> Read( size_t i, size_t j )
		{
			size_t iset  = i % NSET;
			size_t jset  = j % NSET;
			size_t iway  = i / NSET;
			size_t jway  = j / NSET;
			size_t idway = jway * ldway + iway;

			auto &set  = table[ iset ][ jset ];
			auto &lock = locks[ iset ][ jset ];

			/** return values */
			Data<T> KIJ;

			lock.Acquire();
			{
			  auto it = set.find( idway );

			  /** cache hit */
			  if ( it != set.end() )
				{
					KIJ.resize( 1, 1 );
					KIJ[ 0 ] = (*it).second.first;
				}
			}
			lock.Release();

			return KIJ;
			
		}; /** end Read() */


		void Write( size_t i, size_t j, pair<T, double> v )
		{
			size_t iset  = i % NSET;
			size_t jset  = j % NSET;
			size_t iway  = i / NSET;
			size_t jway  = j / NSET;
			size_t idway = jway * ldway + iway;

			auto &set  = table[ iset ][ jset ];
			auto &lock = locks[ iset ][ jset ];

			lock.Acquire();
			{
			  /** cache miss */
			  if ( !set.count( idway ) )
				{
					if ( set.size() < NWAY ) 
					{
						set[ idway ] = v;
					}
					else
					{
						auto target = set.begin();
						for ( auto it = set.begin(); it != set.end(); it ++ )
						{
							/** if this candidate has lower reuse frequency */
              if ( (*it).second.second < (*target).second.second ) target = it;
						}
						if ( target != set.end() )
						{
							if ( (*target).second.second < v.second )
							{
							  set.erase( target );
							  set[ idway ] = v;
							}
						}
					}
				}
			}
			lock.Release();

		}; /** end Write() */

	private:

		size_t n = 0;

		size_t ldway = 0;

		vector<vector<Lock>> locks;

		vector<vector<map<size_t, pair<T, double>>>> table;

}; /** end class Cache2D */






}; /** end namespace hmlp */

#endif /** define CACHE_HPP */
