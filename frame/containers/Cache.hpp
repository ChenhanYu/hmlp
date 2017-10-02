#ifndef CACHE_HPP
#define CACHE_HPP

#include <hmlp_runtime.hpp>
#include <containers/data.hpp>


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

	  hmlp::Data<T> Read( size_t id )
		{
			hmlp::Data<T> ret;

			/** use direct map to find the set */
			auto &set = table[ id % NSET ];

			locks[ id % NSET ].Acquire();
			{
				/** use std::map::find inside the fully associated set */
				auto it = set.find( id );
				if ( it != set.end() ) 
				{
					size_t line_id = (*it).second;
					std::vector<size_t> J( 1, line_id );
					/** cache hit */
					ret = buffer( all_units, J );
					/** increate frequency */
					freq[ line_id ] ++;
				}
			}
			locks[ id % NSET ].Release();

			return ret;
		};

		void Write( size_t id,  hmlp::Data<T> &input )
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

		std::vector<size_t> all_units;

		std::vector<hmlp::Lock> locks;

		std::vector<std::map<size_t, size_t>> table;

		std::vector<size_t> freq;

		hmlp::Data<T> buffer;

};



}; /** end namespace hmlp */

#endif /** define CACHE_HPP */
