// Can be included multiple times.
INSTANCE_FUNCTION(GetDeviceProcAddr);
INSTANCE_FUNCTION(GetPhysicalDeviceMemoryProperties);
INSTANCE_FUNCTION(GetPhysicalDeviceProperties2);
INSTANCE_FUNCTION(GetPhysicalDeviceVideoFormatPropertiesKHR);
INSTANCE_FUNCTION(GetPhysicalDeviceFormatProperties2);
INSTANCE_FUNCTION(GetPhysicalDeviceImageFormatProperties2);
INSTANCE_FUNCTION(GetPhysicalDeviceVideoCapabilitiesKHR);
INSTANCE_FUNCTION(GetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR);
INSTANCE_FUNCTION(GetPhysicalDeviceQueueFamilyProperties);
DEVICE_FUNCTION(QueueSubmit2);
DEVICE_FUNCTION(WaitSemaphores);
DEVICE_FUNCTION(DestroyImageView);
DEVICE_FUNCTION(DestroyImage);
DEVICE_FUNCTION(DestroyBuffer);
DEVICE_FUNCTION(FreeMemory);
DEVICE_FUNCTION(DestroyCommandPool);
DEVICE_FUNCTION(CreateCommandPool);
DEVICE_FUNCTION(AllocateCommandBuffers);
DEVICE_FUNCTION(DestroyPipeline);
DEVICE_FUNCTION(DestroyPipelineLayout);
DEVICE_FUNCTION(DestroyDescriptorSetLayout);
DEVICE_FUNCTION(DestroySemaphore);
DEVICE_FUNCTION(CreateSemaphore);
DEVICE_FUNCTION(DestroyQueryPool);
DEVICE_FUNCTION(AllocateMemory);
DEVICE_FUNCTION(MapMemory);
DEVICE_FUNCTION(CreateBuffer);
DEVICE_FUNCTION(GetBufferMemoryRequirements);
DEVICE_FUNCTION(BindBufferMemory);
DEVICE_FUNCTION(BindImageMemory);
DEVICE_FUNCTION(CreateComputePipelines);
DEVICE_FUNCTION(CreateShaderModule);
DEVICE_FUNCTION(CreateDescriptorSetLayout);
DEVICE_FUNCTION(CreatePipelineLayout);
DEVICE_FUNCTION_FALLBACK(CmdPushDescriptorSetKHR, CmdPushDescriptorSet);
DEVICE_FUNCTION(DestroyShaderModule);
DEVICE_FUNCTION(CreateVideoSessionKHR);
DEVICE_FUNCTION(DestroyVideoSessionKHR);
DEVICE_FUNCTION(GetVideoSessionMemoryRequirementsKHR);
DEVICE_FUNCTION(BindVideoSessionMemoryKHR);
DEVICE_FUNCTION(CreateVideoSessionParametersKHR);
DEVICE_FUNCTION(DestroyVideoSessionParametersKHR);
DEVICE_FUNCTION(GetEncodedVideoSessionParametersKHR);
DEVICE_FUNCTION(CreateImage);
DEVICE_FUNCTION(CreateImageView);
DEVICE_FUNCTION(GetImageMemoryRequirements);
DEVICE_FUNCTION(CreateQueryPool);
DEVICE_FUNCTION(GetQueryPoolResults);
DEVICE_FUNCTION(ResetCommandBuffer);
DEVICE_FUNCTION(BeginCommandBuffer);
DEVICE_FUNCTION(EndCommandBuffer);
DEVICE_FUNCTION(CmdPushConstants);
DEVICE_FUNCTION(CmdDispatch);
DEVICE_FUNCTION(CmdBindPipeline);
DEVICE_FUNCTION(CmdPipelineBarrier2);
DEVICE_FUNCTION(CmdCopyImage);
DEVICE_FUNCTION(CmdBeginVideoCodingKHR);
DEVICE_FUNCTION(CmdEndVideoCodingKHR);
DEVICE_FUNCTION(CmdControlVideoCodingKHR);
DEVICE_FUNCTION(CmdResetQueryPool);
DEVICE_FUNCTION(CmdBeginQuery);
DEVICE_FUNCTION(CmdEndQuery);
DEVICE_FUNCTION(CmdEncodeVideoKHR);
DEVICE_FUNCTION(QueueWaitIdle);
DEVICE_FUNCTION(CmdWriteTimestamp2);
