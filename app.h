#ifndef VK_APP_H
#define VK_APP_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>

struct QueueFamilies {
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;

	bool is_complete() {
		return graphics_family.has_value() && present_family.has_value();
	}
};

struct SwapchainSupport {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> present_modes;
};

class VkApp {

private:
	bool validation_enabled_;

	GLFWwindow* window_;

	vk::Instance instance_;
	vk::DebugUtilsMessengerEXT dbg_messenger_;
	vk::SurfaceKHR surface_;

	vk::PhysicalDevice physical_device_;
	QueueFamilies queue_family_indices_;
	vk::Device device_;
	vk::Queue graphics_queue_;
	vk::Queue present_queue_;

	vk::SurfaceFormatKHR swap_format_;
	vk::PresentModeKHR swap_present_mode_;
	vk::Extent2D swap_extent_;
	vk::SwapchainKHR swapchain_;
	std::vector<vk::Image> swapchain_images_;
	std::vector<vk::ImageView> swapchain_image_views_;

public:
	VkApp(int width, int height, std::string title, bool validation_enabled=false);
	~VkApp();

	void run();

	static VKAPI_ATTR VkBool32 VKAPI_CALL dbg_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT severity,
		VkDebugUtilsMessageTypeFlagsEXT type,
		const VkDebugUtilsMessengerCallbackDataEXT* cb_data,
		void* user_data
	);

private:

	void init_vulkan();

	void init_instance();
	void init_dbg_messenger();
	void init_surface();
	void select_physical_device();
	void init_logical_device();
	void init_swapchain();
	void init_image_views();

	std::vector<const char*> get_required_exts() const noexcept;

	bool is_device_suitable(vk::PhysicalDevice device) const noexcept;
	QueueFamilies get_queue_families(vk::PhysicalDevice device) const noexcept;
	bool supports_extensions(vk::PhysicalDevice device) const noexcept;

	SwapchainSupport get_swapchain_support(vk::PhysicalDevice device) const noexcept;
	vk::SurfaceFormatKHR select_swap_surface_format(std::vector<vk::SurfaceFormatKHR> formats) const noexcept;
	vk::PresentModeKHR select_swap_present_mode(std::vector<vk::PresentModeKHR> modes) const noexcept;
	vk::Extent2D select_swap_extent(vk::SurfaceCapabilitiesKHR capabilities) const noexcept;
};

#endif