
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cstring.h>
#include <lucius/parallel/interface/Vector.h>
#include <lucius/parallel/interface/assert.h>

// Standard Library Includes
#include <cstdint>
#include <cmath>

namespace lucius
{
namespace parallel
{

CUDA_DECORATOR inline size_t strlen(const char* string)
{
    const char* position = string;

    while(*position != '\0')
    {
        ++position;
    }

    return position - string;
}

CUDA_DECORATOR inline void* memcpy(void* dest, const void* src, size_t count)
{
    //TODO: more efficient

    const int8_t* source      = reinterpret_cast<const int8_t*>(src);
          int8_t* destination = reinterpret_cast<      int8_t*>(dest);

    for(size_t i = 0; i < count; ++i)
    {
        destination[i] = source[i];
    }

    return dest;
}

class lldiv_t
{
public:
    long long int quot;
    long long int rem;
};

CUDA_DECORATOR inline lldiv_t div(long long int n, long long int d)
{
    lldiv_t result;

    result.quot = n / d;
    result.rem  = n % d;

    return result;
}

template<typename T>
CUDA_DECORATOR inline T i2a(char* s, T n)
{
    vector<T> digits;

    if(n == 0)
    {
        return 0;
    }

    while(n != 0)
    {
        lldiv_t qr = div(n, 10);

        digits.push_back(qr.rem);

        n = qr.quot;
    }

    for(size_t i = 0; i < digits.size(); ++i)
    {
        s[i] = digits[i] + '0';
    }

    return digits.size();
}

CUDA_DECORATOR inline char* itoa(char* output_buff, long long int num)
{
    char* p = output_buff;

    if(num < 0)
    {
        *p++ = '-';
        num *= -1;
    }
    else if(num == 0)
    {
        *p++ = '0';
    }

    p[i2a(p, num)] = '\0';

    return output_buff;
}

CUDA_DECORATOR inline char* itoa(char* output_buff, unsigned long long int num)
{
    char* p = output_buff;

    if(num == 0)
    {
        *p++ = '0';
    }

    p[i2a(p, num)] = '\0';

    return output_buff;
}

CUDA_DECORATOR inline char* itoh(char* output_buff, long long int num)
{
    char* p = output_buff;

    if(num == 0)
    {
        *p++ = '0';
    }

    while(num > 0)
    {
        long long int current = num & 0xF;
        long long int next    = (num >> 4) & 0xF;
        *p++ = (current <= 9 ? '0' + current : 'A' - 10 + current);
        *p++ = (next    <= 9 ? '0' + next    : 'A' - 10 + next   );
        num = num >> 8;
    }

    *p = '\0';

    return output_buff;
}

CUDA_DECORATOR inline char* dtoa(char* output_buff, double num,
    size_t maxIntegerDigits, size_t maxDecimalDigits)
{
    vector<unsigned int> integerDigits;

    char* p = output_buff;

    // integer component
    double integer = num;

    for(size_t digit = 0; digit < maxIntegerDigits; ++digit)
    {
        double floatDigit = std::floor(std::fmod(integer, 10.0));

        integerDigits.push_back(floatDigit);

        integer = std::floor(integer / 10.0);

        if(integer == 0.0)
        {
            break;
        }
    }

    for(auto digit = integerDigits.rbegin(); digit != integerDigits.rend(); ++digit)
    {
        *(p++) = '0' + *digit;
    }

    *(p++) = '.';

    // decimal component
    double fraction = num - std::floor(num);

    for(size_t digit = 0; digit < maxDecimalDigits; ++digit)
    {
        fraction *= 10.0;

        (*p++) = '0' + static_cast<int>(fraction);

        fraction = fraction - std::floor(fraction);

        if(fraction == 0.0)
        {
            break;
        }
    }

    *(p++) = '\0';

    return output_buff;
}

}
}
