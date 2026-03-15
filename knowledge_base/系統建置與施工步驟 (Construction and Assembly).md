# Technical Construction Guide for Primary Aquaponics Architectures

## 1. Deep Water Culture (DWC) / Raft System Construction Guide

### Section Introduction & Strategic Context
Deep Water Culture (DWC) represents the industrial gold standard for high-turnover leafy green production. From a systems engineering perspective, the strategic advantage of DWC lies in its massive thermal mass; the high water-to-plant ratio provides inherent thermal stability, buffering sensitive root systems against the radical temperature fluctuations common in greenhouse environments. For commercial resilience, a "Dual-Loop" design—as pioneered by OSU—is highly recommended. By utilizing separate aquaculture and hydroponic loops connected via heat exchangers or valves, engineers can isolate systems to treat plant pathogens without exposing fish to phytotoxic compounds, or vice versa.

### System Characteristics & Ideal Use Case
* **Operational Mechanics:** DWC utilizes aerated reservoirs where plants are supported on floating, closed-cell rafts, allowing roots to remain permanently submerged in a nutrient-rich, homogenized solution.
* **Ideal Environment:** Due to the substantial static load of water (8.3 lbs per gallon), DWC is restricted to ground-level installations. It is the premier choice for climate-controlled greenhouses where thermal buffering is a priority.
* **Target Crops:** The constant moisture and oxygen availability favor Lettuce, Spinach, Kale, and Basil. High planting densities in DWC necessitate robust ventilation to prevent foliar fungal infections.

### Required Materials List
* **Reservoirs:** Food-grade containers (e.g., 100-gallon Rubbermaid stock tanks or 4' x 20' lined wooden beds).
* **Rafts:** Closed-cell polystyrene insulation boards; specialized trays with 8-inch hole spacing are standard for leafy greens.
* **Plumbing:** 1.5" to 2" PVC/CPVC piping, bulkhead fittings, and ball valves for drainage.
* **Aeration Suite:** Linear piston or high-output membrane pumps. Calculating for a baseline of 1–2 L/min of air per 100 liters of water is mandatory for low-to-medium stocking densities.
* **Support:** Cinder blocks and opaque covering materials to prevent UV penetration and subsequent algae blooms.

### Step-by-Step Assembly Process
1. **Site Leveling:** Calibrate the reservoir floor to be level within one inch. Failure to do so creates nutrient dead zones and uneven raft buoyancy.
2. **Tank Placement & Support:** Position the main reservoirs on stable ground. For a Dual-Loop setup, ensure the aquaculture loop (fish tanks and filters) is positioned to allow gravity-assisted flow into the hydroponic DWC beds.
3. **Bulkhead & Drainage Installation:** Drill and secure a 1.5" bulkhead and valve at the lowest point of the reservoir to facilitate total system drainage.
4. **Aeration System Deployment:** Install fine-bubble air stones at a density of one per 10 square feet. Connect to a linear piston pump using silicone tubing and **mandatory check valves** to prevent back-siphoning during power outages.
5. **Raft Preparation:** Size rafts to minimize gaps against reservoir walls. Light exposure to the water column must be strictly eliminated to prevent algae competition for nutrients.
6. **Water Filling & Homogenization:** Fill the system. Utilize a magnetic-drive or centrifugal pump to ensure the nutrient solution is thoroughly mixed before planting.
7. **System Cycling & Dechlorination:** Run the system for 48 hours to off-gas chlorine. **Crucial:** Municipal chloramines do not off-gas; you must treat the water with 500mg of Ascorbic Acid (Vitamin C) per 50 gallons or utilize an activated carbon filter before introducing biological life.

### Critical Engineering Tuning
To select the correct air pump (blower), engineers must convert the L/min requirement to Cubic Feet per Minute (CFM), the standard for US commercial hardware. Use the formula: 
$$(Total Liters / 100) \times (1 \text{ to } 2) \times 0.0353 = \text{Required CFM}$$
For a standard 1,000-liter setup, a pump providing at least 0.7 CFM at the system's specific head depth is required. Maintain a stocking density baseline of 1 lb of fish per 10 gallons of water for optimal nutrient balance. DWC provides unmatched stability for leafy greens, but for operations where vertical footprint and root oxygenation are the priorities, the Nutrient Film Technique is the superior architectural choice.

## 2. Nutrient Film Technique (NFT) Construction Guide

### Section Introduction & Strategic Context
The Nutrient Film Technique (NFT) maximizes vertical space by arranging plant channels in tiered or A-frame configurations. Strategically, NFT provides the highest root-zone oxygen exposure of any recirculating system, as the bare roots sit within a shallow, fast-moving film of water. However, the system's low water volume makes it exceptionally sensitive to pump failure. Engineering redundancy—specifically a Dual-Loop strategy to allow for separate aquaculture and plant maintenance—is critical for commercial-grade NFT stability.

### System Characteristics & Ideal Use Case
* **Operational Mechanics:** A shallow stream (film) of nutrient-rich water flows continuously past bare roots in watertight gullies, typically made of UV-protected PVC or specialized gutters.
* **Space Utilization:** NFT is the premier choice for urban farming or greenhouses where increasing yield-per-square-foot through vertical stacking is required.
* **Filtration Requirements:** NFT provides zero mechanical filtration. A separate solids-settling tank (Radial/Swirl Separator) and a dedicated biofilter (sized at 15–20% of the fish tank volume) are non-negotiable.

### Required Materials List
* **Channels:** UV-protected PVC pipes or specialized NFT gutters (maximum 12 meters per run without secondary feed).
* **Support Structure:** Rigid A-frame or tiered racks designed for precise slope maintenance.
* **Plumbing Loop:** High-head submersible or centrifugal pump, 1/2" to 1" delivery manifolds, and 1.5" return PVC pipes.
* **Filtration:** Radial/Swirl separator for heavy solids; biofilter tank with high-surface-area media.
* **Plant Support:** Net pots and rockwool or clay pebble starters.

### Step-by-Step Assembly Process
1. **Support Rack Construction:** Assemble a rigid frame capable of supporting the static weight of the channels and water without sagging, which prevents hazardous ponding.
2. **Channel Mounting & Sloping:** Secure channels to the rack. Calibrate the slope to a ratio of 1:30 to 1:40. For home-scale PVC systems, a slope of **1/2 inch per 10 feet** is the actionable baseline.
3. **Plumbing the Delivery Manifold:** Install the water pump and run delivery lines to the head of each gully. **Engineering Mandate:** For channels exceeding 12 meters, install a secondary nutrient feed halfway along the gully to prevent nitrogen and oxygen depletion.
4. **Integrated Filtration Setup:** Plumb the fish tank overflow to the radial filter, followed by the biofilter, before water enters the NFT manifold.
5. **Flow Rate Calibration:** Adjust valves to achieve a target flow of 1 liter per minute per gully. Rates exceeding 2 L/min typically result in nutritional uptake issues.
6. **Seeding:** Insert seedlings in net pots into the pre-drilled holes, ensuring the base of the pot makes contact with the nutrient film to prevent transplant shock.

### Critical Engineering Tuning
The slope is the primary failure point in NFT systems. While 1:100 is often cited in literature, professional engineering mandates **1:30 to 1:40** to compensate for minor surface irregularities. This steeper gradient ensures a "film" rather than "ponds," which would otherwise lead to anaerobic root rot. While NFT offers high productivity in vertical layouts, media-based systems provide a more robust, multi-functional alternative for heavy-cropping and beginner-scale installations.

## 3. Media-Based Grow Beds (Flood and Drain) Construction Guide

### Section Introduction & Strategic Context
Media-based grow beds are the most resilient aquaponics architecture, functioning as a plant support system, mechanical filter, and biofilter simultaneously. The "Flood and Drain" cycle—regulated by a bell siphon—oxygenates the root zone while hosting critical nitrifying bacteria (*Nitrosomonas* and *Nitrospira*). This architecture is strategically vital for heavy fruiting crops (tomatoes, peppers) that require deep structural anchorage and high nutrient loads.

### System Characteristics & Ideal Use Case
* **Operational Mechanics:** A pump fills the media bed to a maximum height, triggering a bell siphon that rapidly drains the water, pulling atmospheric oxygen down into the media.
* **Multi-functionality:** The porous media acts as the primary site for nitrification and solids capture, often removing the need for separate filtration tanks at low stocking densities.
* **Versatility:** Ideal for fruiting crops. Media beds should be **12 to 14 inches deep** to provide adequate root volume and effective biofiltration.

### Required Materials List
* **Grow Beds:** Food-grade containers 12" deep; 1:1 or 2:1 ratio of grow bed volume to fish tank volume is standard.
* **Grow Media:** Inert, pH-neutral material (1/2" to 3/4" diameter). Expanded clay pebbles or expanded shale are preferred; river rock must be limestone-free.
* **Siphon Components:** Bell siphon kit. **Requirement:** The standpipe height determines the max water level; the media shroud must be significantly wider than the bell to ensure a clean "siphon break."
* **Fish Tank:** Food-grade stock tanks (e.g., Rubbermaid).
* **Pumping System:** Submersible pump.

### Step-by-Step Assembly Process
1. **Foundation Leveling:** Place the fish tank and grow bed stands on a stable surface (cinder blocks). Elevate the fish tank to the highest point if using a gravity-return design.
2. **Grow Bed Plumbing:** Drill for the standpipe and install the bell siphon assembly using a Uniseal® or bulkhead fitting.
3. **Media Preparation:** Thoroughly wash media to remove dust. Conduct a **vinegar test**: if the media fizzes, it contains limestone and will spike the pH; it must be rejected.
4. **Media Filling:** Fill the beds to within 1 inch of the maximum water height (the "dry zone") to prevent algae growth and pests on the media surface.
5. **Water Pump Installation:** Position the pump in the sump or fish tank. Plumb to the grow bed using PVC or flexible tubing.
6. **Siphon Tuning:** Adjust the pump's flow. The siphon must "fire" at the top of the standpipe and "break" once the bed is empty. If the flow is too low, the siphon won't trigger; if too high, it won't break.
7. **Biological Start-up:** Cycle the system. Neutralize chloramines using Ascorbic Acid (500mg per 50 gal) to protect the developing bacterial colonies.

### Critical Engineering Tuning
To calculate the required pump GPH, you must account for the fact that media occupies approximately 60% of the bed's volume. Use the following formula for displaced water volume: 
$$Pump GPH = (Total Bed Volume \times 0.40) + (Fish Tank Volume \times 2)$$
Ensure the media shroud is perforated and clear of debris; it must be wider than the bell to allow air to enter the siphon at the end of the cycle, facilitating the "break." For long-term health, maintain a stocking density of 1 lb of fish per 10 gallons, a target pH of 6.8–7.0, and dissolved oxygen levels above 5 ppm.