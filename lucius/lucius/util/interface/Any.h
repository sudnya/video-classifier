/*! \file   Any.h
    \date   May 14, 2016
    \author Gregory Diamos
    \brief  The header file for the Any class.
*/

#pragma once

// Standard Library Includes
#include <memory>

namespace lucius
{
namespace util
{

/*! \brief A value container for any assignable type, modeled after boost::any. */
class Any
{
public:
    Any()
    {

    }

    ~Any()
    {

    }

public:
    Any(const Any& any)
    {
        if(any.container)
        {
            container.reset(any.container->clone());
        }
    }

    template <typename T>
    Any(const T& value)
    : container(std::make_unique<Container<typename std::remove_const<T>::type>>(value))
    {

    }

    template <typename T>
    Any& operator=(const T& value)
    {
        container = std::make_unique<Container<typename std::remove_const<T>::type>>(value);

        return *this;
    }

    Any& operator=(const Any& any)
    {
        if(&any == this)
        {
            return *this;
        }

        if(any.container)
        {
            container.reset(any.container->clone());
        }
        else
        {
            container.reset(nullptr);
        }

        return *this;
    }

public:
    template <typename T>
    const T& get() const
    {
        auto* valueContainer = dynamic_cast<
            const Container<typename std::remove_const<T>::type>*>(container.get());

        if(valueContainer == nullptr)
        {
            throw std::bad_cast();
        }

        return valueContainer->value;
    }

    template <typename T>
    T& get()
    {
        auto* valueContainer = dynamic_cast<
            Container<typename std::remove_const<T>::type>*>(container.get());

        if(valueContainer == nullptr)
        {
            throw std::bad_cast();
        }

        return valueContainer->value;
    }

public:
    bool empty() const
    {
        return !static_cast<bool>(container);
    }

private:
    class GenericContainer
    {
    public:
        virtual ~GenericContainer()
        {

        }

    public:
        virtual GenericContainer* clone() const = 0;
    };

    template<typename T>
    class Container : public GenericContainer
    {
    public:
        Container(const T& value)
        : value(value)
        {

        }

    public:
        virtual GenericContainer* clone() const
        {
            return new Container<T>(*this);
        }

    public:
        T value;
    };

private:
    std::unique_ptr<GenericContainer> container;

};

}
}

