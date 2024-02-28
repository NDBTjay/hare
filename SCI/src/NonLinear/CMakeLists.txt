add_library(SCI-NonLinear INTERFACE)
target_link_libraries(SCI-NonLinear
	INTERFACE SCI-Millionaire gemini SEAL::seal
)
