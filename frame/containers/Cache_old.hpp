#ifndef CACHE_HPP
#define CACHE_HPP


#include <unordered_map>

#include <hmlp_runtime.hpp>
#include <Data.hpp>

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


template<typename T>
class Cache2D
{
  public:


    Cache2D( const uint64_t num_of_row_sets, const uint64_t num_of_col_sets,
        const uint64_t num_of_ways_per_set, const uint64_t vector_width, 
        const uint64_t height, const uint64_t wdith )
      : num_or_row_sets_( num_of_row_sets ), 
        num_of_col_sets_( num_of_col_sets ),
        num_of_ways_per_set_( num_of_ways_per_set ),
        vector_width_( vector_width ),
        height_( height ),
        width_( width ),
        ldway_( height / num_of_row_sets + 1 )
    {
      set_locks_2d_.resize( num_of_row_sets_ * num_of_col_sets_ );
      set_hashtables_2d_.resize( num_of_row_sets_ * num_of_col_sets_ );
      /* Overall how many elements in type T are held in the cache? */
      cached_data_2d_.resize( num_of_row_sets_ * num_of_col_sets_ );
      for ( auto & data : cache_data_2d_ )
      {
        data.resize( num_of_ways_per_set_ * vector_width_ );
      }
    };

    hmlpError_t read( uint64_t i, uint64_t j, std::vector<T>& line, std::vector<uint64_t>& row_indices )
    {
      if ( i >= height_ || j >= width_ )
      {
        return HMLP_ERROR_INVALID_VALUE;
      }
      /* Compute which set (i, j) belongs to. */
      uint64_t iset  = ( i / vector_width_ ) % num_of_row_sets_;
      uint64_t jset  = j % num_of_col_sets;
      /* Compute the key of this cache line. */
      uint64_t first_row_index = ( i / vector_width_ ) * vector_width_;
      uint64_t key = j * height + first_row_index;
      uint32_t hit_index_in_vector = i % vector_width_;
      /* Acquire the hashtable and the lock to grant exclusive access. */
      auto & set  = set_hashtables_2d_[ jset * num_of_row_sets + iset ];
      auto & lock = set_locks_2d_[ jset * num_of_row_sets + iset ];
      auto & data = cached_data_2d_[ jset * num_of_row_sets + iset ];
      /* Clear the cache line and tbe row_indices. */
      line.clear();
      row_indices.clear();
      /* Now resize row_indices with the proper indices. */
      for ( uint64_t i = first_row_index; i < first_row_index + vector_width; i ++ )
      {
        if ( i < height_ ) row_indices.push_back( i );
      }

      lock.Acquire();
      {
        /* Cache hit! The line is cached. */
        if ( set.count( key ) )
        {
          /* This offset should be small than num_of_way_per_set_. */
          uint64_t offset_to_cached_data = set[ key ] * vector_width_;
          line.insert( line.begin(), 
              data.begin() + offset_to_cached_data, 
              data.begin() + offset_to_cached_data + vector_width_ );
        }
      }
      lock.Release();

      /* Return with no error. */
      return HMLP_ERROR_SUCCESS;
    };

    hmlpError_t write( const std::vector<uint64_t>& row_indices, uint64_t j, const std::vector<uint64_t>& line )
    {
      if ( row_indices.size() != line.size() )
      {
        return HMLP_ERROR_INVALID_VALUE;
      }
      if ( row_indices.size() == 0 )
      {
        return HMLP_ERROR_SUCCESS;
      }
      /* Access the first index */
      uint64_t i = row_indieces[ 0 ];
      /* Compute which set (i, j) belongs to. */
      uint64_t iset  = ( i / vector_width_ ) % num_of_row_sets_;
      uint64_t jset  = j % num_of_col_sets;
      /* Compute the key of this cache line. */
      uint64_t first_row_index = ( i / vector_width_ ) * vector_width_;
      uint64_t key = j * height + first_row_index;
      uint32_t hit_index_in_vector = i % vector_width_;
      /* Check if the first index is valid? */
      if ( hit_index_in_vector != 0 )
      {
        return HMLP_ERROR_INVALID_VALUE;
      }
      /* Check if each index is valid? */
      for ( uint64_t it = 0; it < row_indices.size(); it ++ )
      {
        if ( row_indices[ it ] % vector_width_ != it )
        {
          return HMLP_ERROR_INVALID_VALUE;
        }
      }
      /* Acquire the hashtable and the lock to grant exclusive access. */
      auto & set  = set_hashtables_2d_[ jset * num_of_row_sets + iset ];
      auto & lock = set_locks_2d_[ jset * num_of_row_sets + iset ];
      auto & data = cached_data_2d_[ jset * num_of_row_sets + iset ];

      lock.Acquire();
      {
        /* Cache hit! The line is cached. */
        if ( set.count( key ) == 0 )
        {
          if ( set.size() < num_of_way_per_set_ )
          {
            set[ key ] = set.size();
          }
          else
          {
          }
        }


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
    };

  private:

    const uint64_t num_of_row_sets_ = 0;
    const uint64_t num_of_col_sets_ = 0;
    const uint64_t num_of_ways_per_set_ = 0;
    const uint64_t vector_width_ = 1;
    const uint64_t height_ = 0;
    const uint64_t width_ = 0;
    const uint64_t ldway_ = 0;

    std::vector<Lock> set_locks_2d_;

    std::vector<unordered_map<uint64_t, uint64_t>> set_hashtables_2d_;

    std::vector<std::vector<T>> cached_data_2d_;

}; /* end class Cache2D */
}; /* end namespace hmlp */

#endif /* define CACHE_HPP */
