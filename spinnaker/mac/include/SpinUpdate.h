//=============================================================================
// Copyright (c) 2001-2021 FLIR Systems, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

#ifndef FLIR_SPIN_UPDATE_H
#define FLIR_SPIN_UPDATE_H

#include <string>
#include <vector>

/* Function declaration modifiers */
#if defined(_WIN32)
#ifndef SPINUPDATE_NO_DECLSPEC_STATEMENTS
#ifdef SPINUPDATE_EXPORTS
#define SPINUPDATE_IMPORT_EXPORT __declspec(dllexport)
#else
#define SPINUPDATE_IMPORT_EXPORT __declspec(dllimport)
#endif
#else
#define SPINUPDATE_IMPORT_EXPORT
#endif /* #  ifndef GCTLI_NO_DECLSPEC_STATEMENTS */
#ifndef _M_X64
#define SPINUPDATE_CALLTYPE __stdcall
#else
#define SPINUPDATE_CALLTYPE /* default */
#endif
#ifndef EXTERN_C
#define EXTERN_C extern "C"
#endif

#elif defined(__GNUC__) && (__GNUC__ >= 4) && (defined(__linux__) || defined(__APPLE__))
#define SPINUPDATE_IMPORT_EXPORT __attribute__((visibility("default")))
#ifndef __x86_64
#define SPINUPDATE_CALLTYPE __attribute__((stdcall))
#else
#define SPINUPDATE_CALLTYPE /* default */
#endif
#ifndef EXTERN_C
#define EXTERN_C extern "C"
#endif

#else
#error Unknown platform, file needs adaption
#endif

#ifdef __cplusplus
extern "C"
{
    namespace SpinUpdate
    {
#endif

#define SPINUPDATE_API SPINUPDATE_IMPORT_EXPORT

        /**
         * Updates the firmware for the device.
         *
         * @param numArgs Number of strings pointed to by argv
         * @param argList Pointer to list of string options for the firmware update
         *
         * @return 0 for success, otherwise non zero for failures.
         *
         * Typical usage for updating is as follows:
         *  -R{serial number} [-{options} ..] {firmware zim file}
         *  -R{serial number} -UU -B {firmware zim file}
         *
         * Option definitions:
         *       -B = Reboots the camera after the update has completed.
         *              If this argument is not provided, a manual power cycle will be required.
         *       -A = Updates individual portions of the firmware in flash.
         *            The code section of camera at location 0xFF08000 will be updated.
         *                 ./sample_app -AFF080000 camera.zim
         *       -U = Downgrade the firmware. Multiple U's can be used to overwrite the ROM header.
         *       -F = Force program and EEprom reload.
         *       -R = Enter a regular expression for camera serial match.
         *              For example: ./sample_app -R.* camera.zim
         *              Results in matching any camera serial
         *       -P = Checks the progress of the updater.
         *       -epromsave = Save the content of the EEprom to a file.
         */
        SPINUPDATE_API int UpdateFirmwareConsole(unsigned int numArgs, char** argList);
        SPINUPDATE_API int UpdateFirmwareGUI(std::string args);
        SPINUPDATE_API int UpdateFirmware(std::vector<std::string> args);

        SPINUPDATE_API typedef int (*UpdatorMessageCallback)(const char* message);
        SPINUPDATE_API typedef int (
            *UpdatorProgressCallback)(const char* action, unsigned int address, int globalPercent, int currPercent);

        SPINUPDATE_API void SetMessageCallback(UpdatorMessageCallback messageCallbackFunction);
        SPINUPDATE_API void SetProgressCallback(UpdatorProgressCallback progressCallbackFunction);

        SPINUPDATE_API const char* GetErrorMessage();

#ifdef __cplusplus
    } /* end of namespace SpinUpdate */
} /* end of extern "C" */
#endif

#endif // FLIR_SPIN_UPDATE_H