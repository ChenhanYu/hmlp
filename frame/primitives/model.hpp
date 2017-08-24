#ifndef MODEL_HPP
#define MODEL_HPP

namespace hmlp
{

template<typename T>
class Classification : public VirtualModel
{
  public:

};


template<typename T>
class Regression : public VirtualModel
{
  public:


};

template<typename T>
class VirtualModel
{
  public:

    VirtualModel() {};

    virtual Fit()

    virtual Predict();

  private:

};

}; /** end namespace hmlp */


#endif /** define MODEL_HPP */
