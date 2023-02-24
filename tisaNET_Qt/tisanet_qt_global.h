#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(TISANET_QT_LIB)
#  define TISANET_QT_EXPORT Q_DECL_EXPORT
# else
#  define TISANET_QT_EXPORT Q_DECL_IMPORT
# endif
#else
# define TISANET_QT_EXPORT
#endif
